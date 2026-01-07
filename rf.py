"""
Class declarations and reading functions
required to unpickle trained RandomForest models

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""

# Global imports
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import UnivariateSpline

def vert_aggregation(radar_data, vert_weights, grp_vertical, 
                  visib_weight = True, visib = None):
    """
    Performs vertical aggregation of radar observations aloft to the ground
    using a weighted average. Categorical variables such as 'RADAR',
    'HYDRO', 'TCOUNT', will be assigned dummy variables and these dummy
    variables will be aggregated, resulting in columns such as RADAR_propA
    giving the weighted proportion of radar observation aloft that were
    obtained with the Albis radar
    
    Parameters
    ----------
    radar_data : Pandas DataFrame
        A Pandas DataFrame containing all required input features aloft as
        explained in the rf.py module 
    vert_weights : np.array of float
        vertical weights to use for every observation in radar, must have
        the same len as radar_data
    grp_vertical : np.array of int
        grouping index for the vertical aggregation. It must have the same
        len as radar_data. All observations corresponding to the same
        timestep must have the same label
    visib_weight: bool
        if True the input features will be weighted by the visibility
        when doing the vertical aggregation to the ground
    visib : np array
        visibily of every observation, required only if visib_weight = True
    """    
    if visib_weight and not np.any(visib is None):
        vert_weights = vert_weights * visib / 100.
    else:
        vert_weights = pd.Series(vert_weights, index=radar_data.index)
            
    X =  pd.DataFrame()  # output
    sum_wvisib = vert_weights.groupby(grp_vertical).sum()

    for v in radar_data.columns:
        if v not in ['RADAR','HYDRO','TCOUNT']:
            X[v] = (radar_data[v] * vert_weights).groupby(grp_vertical).sum() / sum_wvisib
        else:
            # For these variables we aggregate dummy variables
            vals = np.unique(radar_data[v])
            for val in vals:
                X[v+'_prop_'+str(val)] = (((radar_data[v] == val).astype(int) * vert_weights).
                        groupby(grp_vertical).sum() / sum_wvisib)
    return X
##################
# Add here all classes/functions used in the construction of the pickled 
# instances
##################
    
def _polyfit_no_inter(x,y, degree):
    """linear regression with zero intercept"""
    X = []
    for i in range(1,degree+1):
        X.append(x**i)
    X = np.array(X).T
    p, _, _, _ = np.linalg.lstsq(X, y[:,None])
    p = np.insert(p,0,0) # Add zero intercept at beginning for compatibility with polyval
    return p[::-1] # Reverse because that's how it is in polyfit (high degree first)
    
class RandomForestRegressorBC(RandomForestRegressor):
    '''
    This is an extension of the RandomForestRegressor regressor class of
    sklearn that does additional bias correction, is able
    to apply a rounding function to the outputs on the fly and adds a 
    bit of metadata:
    
        *bctype* : type of bias correction method
        *variables* : name of input features
        *beta* : weighting factor in vertical aggregation
        *degree* : order of the polyfit used in some bias-correction methods
        *metadata* : configuration setup used to train this model
    
    For *bc_type* tHe available methods are currently "raw":
    simple linear fit between prediction and observation, "cdf": linear fit
    between sorted predictions and sorted observations and "spline" :
    spline fit between sorted predictions and sorted observations. Any
    new method should be added in this class in order to be used.
    
    For any information regarding the sklearn parent class see
    
    https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/ensemble/_forest.py#L1150
    '''
    def __init__(self, 
                 beta,
                 degree = 1, 
                 bctype = 'cdf',
                 metadata = {},
                 n_estimators=100,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators = n_estimators,
                         criterion = criterion,
                         max_depth = max_depth,
                         min_samples_split = min_samples_split,
                         min_samples_leaf = min_samples_leaf,
                         min_weight_fraction_leaf = min_weight_fraction_leaf,
                         max_features = max_features,
                         max_leaf_nodes = max_leaf_nodes,
                         min_impurity_decrease = min_impurity_decrease,
                         bootstrap = bootstrap,
                         oob_score = oob_score,
                         n_jobs = n_jobs,
                         random_state = random_state,
                         verbose = verbose,
                         warm_start = warm_start)
        
        self.degree = degree
        self.bctype = bctype
        self.beta = beta
        self.metadata = metadata

    def fit(self, X,y, grp_vertical, heights = None, visib_weighting = False, sample_weight = None):
        """
        Fit the RandomForest regressor and estimate the a-posteriori
        bias-correction model.

        This method first performs a vertical aggregation of radar observations
        aloft to ground level using a weighted average. The aggregated features
        are then used to fit the underlying RandomForestRegressor. After fitting,
        a bias-correction function is estimated by comparing model predictions
        to observations.

        Parameters
        ----------
        X : pandas.DataFrame
            Input radar observations aloft. Must contain all features required
            by :func:`vert_aggregation`, including at least the columns
            ``'HEIGHTS'`` and, if ``visib_weighting=True``, ``'VISIB_mean'``.

            Each row corresponds to one radar observation aloft.

        y : array-like of shape (n_samples,)
            Target values (e.g., ground precipitation measurements). Each value
            corresponds to one vertically aggregated profile defined by
            ``grp_vertical``.

        grp_vertical : array-like of shape (n_rows,)
            Grouping index defining which rows of ``X`` belong to the same
            vertical profile / timestep. All rows with the same label are
            aggregated together.

        heights : array-like of shape (n_rows,), optional
            Heights (in meters) of the radar observations. If not provided,
            they are read from ``X['HEIGHTS']``.

        visib_weighting : bool, default=False
            If True, vertical weights are additionally scaled by the visibility
            stored in ``X['VISIB_mean']``.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights passed to the underlying RandomForestRegressor.
            If None, all samples are weighted equally.

        Returns
        -------
        self : RandomForestRegressorBC
            Fitted estimator with learned bias-correction parameters stored
            in ``self.p``.
        """
        
        try:
            heights = X['HEIGHT']
        except:
            raise
        vweights = 10**(self.beta *
                            (heights/1000.)) # vert. weights
        
        if visib_weighting:
            visib = X['VISIB_mean']
        else:
            visib = None
            
        features_VERT_AGG = vert_aggregation(X, 
                                vweights, grp_vertical,
                                visib_weighting,
                                visib)
        
        super().fit(features_VERT_AGG, y, sample_weight)
        y_pred = super().predict(features_VERT_AGG)
        if self.bctype in ['cdf','raw']:
            if self.bctype == 'cdf':
                x_ = np.sort(y_pred)
                y_ = np.sort(y)
            elif self.bctype == 'raw':
                x_ = y_pred
                y_ = y
            self.p = _polyfit_no_inter(x_,y_,self.degree)
        elif self.bctype == 'spline':
            x_ = np.sort(y_pred)
            y_ = np.sort(y)
            _,idx = np.unique(x_, return_index = True)
            self.p = UnivariateSpline(x_[idx], y_[idx])
        else:
            self.p = 1
            
        return 
    
    def predict(self, X, grp_vertical, heights = None, visib_weighting = False, bc =True, round_func = None   ):
        """
        Predict regression targets for new data, with optional bias correction
        and output rounding.

        This method first performs vertical aggregation of radar observations
        using the same procedure as during training. Predictions are then
        generated using the fitted RandomForestRegressor and optionally
        corrected using the learned bias-correction function.

        Parameters
        ----------
        X : pandas.DataFrame
            Input radar observations aloft. Must contain the same features
            used during training, including ``'HEIGHTS'`` and optionally
            ``'VISIB_mean'``.

        grp_vertical : array-like of shape (n_rows,)
            Grouping index defining vertical profiles to be aggregated.
            Must correspond to the rows of ``X``.

        heights : array-like of shape (n_rows,), optional
            Heights (in meters) of the radar observations. If not provided,
            they are read from ``X['HEIGHTS']``.

        visib_weighting : bool, default=False
            If True, visibility weighting is applied during vertical aggregation.

        bc : bool, default=True
            If True, apply the bias-correction function learned during fitting.
            If False, return raw RandomForest predictions.

        round_func : callable, optional
            Function applied element-wise to the final predictions
            (e.g. discretization or lookup-table mapping). If None,
            the identity function is used.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted target values after vertical aggregation, optional
            bias correction, and rounding. Negative values are clipped to zero.
        """
        try:
            heights = X['HEIGHT']
        except:
            raise
        vweights = 10**(self.beta *
                            (heights/1000.)) # vert. weights
        
        if visib_weighting:
            visib = X['VISIB_mean']
        else:
            visib = None
            
        features_VERT_AGG = vert_aggregation(X, 
                                vweights, grp_vertical,
                                visib_weighting,
                                visib)
        
        pred = super().predict(features_VERT_AGG)
        
        if round_func is None:
            def round_func(x):
                return x
        
        def func(x):
            return x
        if bc:
            if self.bctype in ['cdf','raw']:
                def func(x):
                    return np.polyval(self.p,x)
            elif self.bctype == 'spline':
                def func(x):
                    return self.p(x)
        out = func(pred)
        out[out < 0] = 0
        return round_func(out)

    def save(self, path):
        """
        Save the RandomForestRegressorBC instance to disk.

        Parameters
        ----------
        path : str
            File path to save the model (e.g. 'rf_bc_model.pkl').
        """
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a RandomForestRegressorBC instance from disk.

        Parameters
        ----------
        path : str
            File path from which to load the model.

        Returns
        -------
        model : RandomForestRegressorBC
            The loaded model instance.
        """
        # Use your custom unpickler to ensure class is correctly recognized
        with open(path, "rb") as f:
            try:
                # joblib.load internally uses pickle
                model = joblib.load(f)
            except Exception:
                # fallback for custom unpickler if needed
                f.seek(0)
                model = MyCustomUnpickler(f).load()
        return model
    
##################
        
class MyCustomUnpickler(pickle.Unpickler):
    """
    This is an extension of the pickle Unpickler that handles the 
    bookeeeping references to the RandomForestRegressorBC class
    """
    import __main__
    __main__.RandomForestRegressorBC = RandomForestRegressorBC
    def find_class(self, module, name):
        print(module,name)
        return super().find_class(module, name)
    
    