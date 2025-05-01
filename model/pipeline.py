# This file contains the complete pipeline usng transformation i
# classes and modeling algorithm

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import XGBClassifier

from patientsurvival_model.config.core import config
from patientsurvival_model.processing.features import embarkImputer
from patientsurvival_model.processing.features import Mapper
from patientsurvival_model.processing.features import age_col_tfr

patientsurvival_pipe=Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.model_config_.embarked_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', XGBoostClassifier(n_estimators=config.model_config_.n_estimators, 
                                         max_depth=config.model_config_.max_depth, 
                                         max_features=config.model_config_.max_depth,
                                         random_state=config.model_config_.random_state))
          
     ])
