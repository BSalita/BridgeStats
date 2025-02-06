
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)
def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

import pandas as pd
import polars as pl
import os
from fastai.tabular.all import nn, load_learner, tabular_learner, cont_cat_split, TabularDataLoaders, TabularPandas, CategoryBlock, RegressionBlock, Categorify, FillMissing, Normalize, EarlyStoppingCallback, RandomSplitter, range_of, MSELossFlat, L1LossFlat, rmse, accuracy
import time
from datetime import datetime


# Function to calculate the total input size
def calculate_input_size(learn):
    emb_szs = {name: learn.model.embeds[i].embedding_dim for i, name in enumerate(learn.dls.cat_names)}
    total_input_size = sum(emb_szs.values()) + len(learn.dls.cont_names)
    return total_input_size

# Function to define optimal layer sizes
def define_layer_sizes(input_size, num_layers=3, shrink_factor=2):
    layer_sizes = [input_size]
    for i in range(1, num_layers):
        layer_sizes.append(layer_sizes[-1] // shrink_factor)
    return layer_sizes



def train_classifier(df, y_names, cat_names, cont_names, procs=None, valid_pct=0.2, seed=42, bs=1024*5, layers=None, epochs=3, device='cuda', monitor='valid_loss', min_delta=0.001, patience=3):
    t = time.time()
    print_to_log_info(f"{y_names=} {cat_names=} {cont_names} {valid_pct=} {seed=} {bs=} {layers=} {epochs=} {device=} {monitor=} {min_delta=} {patience=}")

    splits_ilocs = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df))
    print_to_log_info(len(splits_ilocs[0]), len(splits_ilocs[1]))    
    for y_name in y_names:
        train_uniques = df.iloc[splits_ilocs[0]][y_name].unique()
        valid_values = df.iloc[splits_ilocs[1]][y_name].values
        valid_uniques = df.iloc[splits_ilocs[1]][y_name].unique()
        values_in_validation_but_not_in_training = set(valid_uniques).difference(train_uniques)
        #new_train_splits = splits_ilocs[0]+[iloc for iloc,value in zip(splits_ilocs[1],valid_values) if value in values_in_validation_but_not_in_training]
        new_valid_splits = [iloc for iloc,value in zip(splits_ilocs[1],valid_values) if value not in values_in_validation_but_not_in_training]
        splits_ilocs = (splits_ilocs[0], new_valid_splits)
        print_to_log_info(values_in_validation_but_not_in_training, len(splits_ilocs[0]), len(splits_ilocs[1]))    

    to = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=y_names,
        splits=splits_ilocs,
        #num_workers=10,
        y_block=CategoryBlock()
    )
    
    assert set(to.valid.y).difference(to.train.y) == set(), f"validation set has classes which are not in the training set:{set(to.valid.y).difference(to.train.y)}"
    # Create a DataLoader
    dls = to.dataloaders(bs=bs, device=device) # cpu or cuda

    # determine layers
    learn = tabular_learner(dls, metrics=accuracy)

    # Calculate the total input size
    input_size = calculate_input_size(learn)

    # Define the optimal layer sizes
    recommended_layers = define_layer_sizes(input_size)
    print(f"Recommended layer sizes: {recommended_layers}")
    
    if layers is None:
        layers = recommended_layers
        print(f"Using recommended layer sizes: {layers}")
    else:
        print(f"Using provided layer sizes: {layers}")

    # Update the learner with the defined layer sizes
    learn = tabular_learner(dls, layers=layers, metrics=accuracy)

    # Train the model
    learn.fit_one_cycle(epochs, cbs=EarlyStoppingCallback(monitor=monitor, min_delta=min_delta, patience=patience)) # 1 or 2 epochs is often enough to get a good accuracy for large datasets
    print_to_log_info('train_classifier time:', time.time()-t)
    return learn



def train_regression(df, y_names, cat_names, cont_names, procs=None, valid_pct=0.2, seed=42, bs=1024*5, layers=None, epochs=3, device='cuda', monitor='valid_loss', min_delta=0.001, patience=3, y_range=(0,1)):
    """
    Train a tabular model for regression.
    """
    t = time.time()
    print_to_log_info(f"{y_names=} {cat_names=} {cont_names} {valid_pct=} {seed=} {bs=} {layers=} {epochs=} {device=} {monitor=} {min_delta=} {patience=} {y_range=}")
    # todo: check that y_names is numeric, not category.

    splits_ilocs = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df))
    print_to_log_info(len(splits_ilocs[0]), len(splits_ilocs[1]))    

    to = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=y_names,
        splits=splits_ilocs,
        #num_workers=10,
        y_block=RegressionBlock()
        )
    
    # Create a DataLoader
    dls = to.dataloaders(bs=bs, device=device) # cpu or cuda

    # determine layers
    learn = tabular_learner(dls, metrics=rmse, y_range=y_range, loss_func=MSELossFlat()) # todo: could try loss_func=L1LossFlat or MSELossFlat

    # Calculate the total input size
    input_size = calculate_input_size(learn)

    # Define the optimal layer sizes
    recommended_layers = define_layer_sizes(input_size)
    print(f"Recommended layer sizes: {recommended_layers}")
    
    if layers is None:
        layers = recommended_layers
        print(f"Using recommended layer sizes: {layers}")
    else:
        print(f"Using provided layer sizes: {layers}")

    # Update the learner with the defined layer sizes
    learn = tabular_learner(dls, layers=layers, metrics=rmse, y_range=y_range, loss_func=MSELossFlat()) # todo: could try loss_func=L1LossFlat or MSELossFlat

    # Use mixed precision training. slower and error.
    # error: Can't get attribute 'AMPMode' on <module 'fastai.callback.fp16'
    #learn.to_fp16() # to_fp32() or to_bf16()
    
    # Use one cycle policy for training with early stopping
    learn.fit_one_cycle(epochs, cbs=EarlyStoppingCallback(monitor=monitor, min_delta=min_delta, patience=patience)) # todo: experiment with using lr_max?
    print_to_log_info('train_regression time:', time.time()-t)
    return learn


def save_model(learn, f):
    t = time.time()
    learn.export(f)
    print_to_log_info('save_model time:', time.time()-t)


def load_model(f):
    t = time.time()
    learn = load_learner(f)
    print_to_log_info('load_model time:', time.time()-t)
    return learn


def make_predictions(f, data):
    """
    Make predictions using a trained tabular model.
    """
    learn = load_learner(f)
    return get_predictions(learn, data)


# doesn't seem to work as train_df dtypes are all object.(?)
# # Function to compare columns and dtypes
# def compare_columns_and_dtypes(train_df, infer_df):
#     train_columns = set(train_df.columns)
#     infer_columns = set(infer_df.columns)

#     # Compare columns
#     missing_in_infer = train_columns - infer_columns
#     extra_in_infer = infer_columns - train_columns

#     if missing_in_infer:
#         print(f"Columns in train_df but not in infer_df: {missing_in_infer}")
#     if extra_in_infer:
#         print(f"Columns in infer_df but not in train_df: {extra_in_infer}")

#     # Compare data types of matching columns
#     common_columns = train_columns & infer_columns
#     for col in common_columns:
#         train_dtype = train_df[col].dtype
#         infer_dtype = infer_df[col].dtype
#         if train_dtype != infer_dtype:
#             print(f"Column '{col}' has different dtypes: train_df ({train_dtype}), infer_df ({infer_dtype})")
#             infer_df[col] = infer_df[col].astype(train_dtype)
#             infer_dtype = infer_df[col].dtype
#             assert train_dtype == infer_dtype, f"Column '{col}' still differs in dtype: train_df ({train_dtype}), infer_df ({infer_dtype})"


def get_predictions(learn, df, y_names=None, device='cpu'):
    """
    Perform inference using a trained model.
    
    learn: Trained Fastai learner
    inference_data: DataFrame containing the inference data
    """
    t = time.time()

    if False: #logger.isEnabledFor(logging.DEBUG):
        df[learn.dls.train.x_names].info(verbose=True)
        if y_names:
            df[y_names].info(verbose=True)
            
    assert set(learn.dls.train.x_names).difference(df.columns) == set(), f"df is missing column names which are in the model's training set:{set(learn.dls.train.x_names).difference(df.columns)}"
    # Retrieve y_names from the dataloader
    if y_names is None:
        y_names = learn.dls.y_names # todo: or should it be learn.dls.train.y_names?
    assert len(y_names) == 1, 'Only one target variable is supported.'
    y_name = y_names[0]
    del y_names

    invalid_dtypes = []
    for col in df.columns:
        if df[col].dtype.name not in ['category','datetime64[us]','object','string']: # dtypes which will be converted by fastai. Otherwise must be a torch compatible dtype.
            if df[col].dtype.name not in ['float64', 'float32', 'float16', 'complex64', 'complex128', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'bool']:
                print_to_log_info(f"Warning: {col} is neither a fastai dtype nor a torch compatible dtype: {df[col].dtype.name}. Convert to a supported dtype")
                invalid_dtypes.append((col, df[col].dtype.name))
    assert invalid_dtypes == [], f"Unsuported (col,dtype): {invalid_dtypes}"

    assert not df.empty, 'No data to make inferences on.'

    #compare_columns_and_dtypes(learn.dls.train.items, df)

    #compare_columns_and_dtypes(learn.dls.valid.items, df)

    if df[y_name].dtype.name in ['object','string']:
        df[y_name] = pd.Categorical(df[y_name],categories=learn.dls.vocab)
    if df[y_name].dtype.name in ['category']: # assumes object is a string and will be converted to category

        train_uniques = learn.dls.vocab
        values_in_test_but_not_in_training = df[y_name][~df[y_name].isin(train_uniques)]
        print_to_log_info(df.loc[values_in_test_but_not_in_training.index])
        print_to_log_info(f'Warning: {y_name} contains values which are missing in training set:',values_in_test_but_not_in_training)
        df = df.drop(values_in_test_but_not_in_training.index)

        # Create the dictionary with class codes as keys and class labels as values
        pred_code_to_actual_code_d = {df[y_name].cat.categories[code]:code for code in df[y_name].cat.codes.tolist()}
        print_to_log_info('pred_code_to_actual_code_d:',pred_code_to_actual_code_d)

        # Create a dataloader for the inference data
        # Compare dtypes
        dtypes_df1 = dict(zip(df.columns, df.dtypes))
        dtypes_df2 = dict(zip(learn.dls.xs.columns, learn.dls.xs.dtypes))

        # Find columns with different dtypes
        different_dtypes = {col: (dtypes_df1[col], dtypes_df2[col]) for col in dtypes_df1 if col in dtypes_df2 and dtypes_df1[col] != dtypes_df2[col]}

        # Report differences
        for col, (dtype1, dtype2) in different_dtypes.items():
            print(f"Column '{col}' has different dtypes: {dtype1} in df1, {dtype2} in df2")
        cols = df.columns.tolist()
        #cols.remove('Date')

        dl = learn.dls.test_dl(df[cols])

        # Make predictions on the inference data
        preds, targets = learn.get_preds(dl=dl)#, with_input=True, with_decoded=True)

        # Convert probabilities to class labels
        pred_codes = preds.argmax(dim=1).tolist()
        pred_codes_to_actual_codes = [pred_code_to_actual_code_d.get(train_uniques[code],None) for code in pred_codes] # PASS is temp!!!
        pred_labels = [train_uniques[code] for code in pred_codes]
        
        # True labels
        true_codes = df[y_name].cat.codes
        true_labels = df[y_name]
        
        results = {
            '_'.join([y_name,'Actual']): true_labels,
            '_'.join([y_name,'Actual','Code']): true_codes,
            '_'.join([y_name,'Targets','Code']): targets.squeeze().tolist(),
            '_'.join([y_name,'Pred']): pred_labels,
            '_'.join([y_name,'Pred','Code']): pred_codes_to_actual_codes,
            '_'.join([y_name,'Match','Code']): [pred_code == true_code for pred_code, true_code in zip(pred_codes_to_actual_codes, true_codes)],
            '_'.join([y_name,'Match']): [pred_label == true_label for pred_label, true_label in zip(pred_labels, true_labels)]
        }
    else:

        # Create a dataloader for the inference data
        dl = learn.dls.test_dl(df)

        # Make predictions on the inference data
        preds, targets = learn.get_preds(dl=dl)#, with_input=True, with_decoded=True)

        # Since this is regression, preds and targets are continuous values
        true_values = targets.squeeze().tolist()
        pred_values = preds.squeeze().tolist()

        results = {
            f'{y_name}_Actual': true_values,
            f'{y_name}_Pred': pred_values,
            f'{y_name}_Error': [pred - true for pred, true in zip(pred_values, true_values)],
            f'{y_name}_AbsoluteError': [abs(pred - true) for pred, true in zip(pred_values, true_values)]
        }        
    return pd.DataFrame(results)


# create a test set using date and sample size. current default is 10k samples ge 2024-07-01.
def sample_by_date(df, include_dates='2024-07-01', max_samples=10000):
    include_date = datetime.strptime(include_dates, '%Y-%m-%d') # i'm not getting why datetime.datetime.strptime isn't working here but the only thing that works elsewhere?

    date_filter = df['Date'] >= include_date
        
    return df.filter(~date_filter), df.filter(date_filter).sample(n=max_samples) if max_samples < date_filter.sum() else df.filter(date_filter)


# Calculate feature importance
def find_first_linear_layer(module):
    if isinstance(module, nn.Linear):
        return module
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for layer in module:
            found = find_first_linear_layer(layer)
            if found:
                return found
    elif hasattr(module, 'children'):
        for layer in module.children():
            found = find_first_linear_layer(layer)
            if found:
                return found
    return None


def get_feature_importance(learn):
    importance = {}

    # Find the first linear layer in the model
    linear_layer = find_first_linear_layer(learn.model)
    if linear_layer is None:
        raise ValueError("No linear layer found in the model.")
    
    # Get the absolute mean of the weights across the input features
    weights = linear_layer.weight.abs().mean(dim=0)

    # Get all feature names
    all_columns = learn.dls.train_ds.items.columns.tolist()
    feature_names = [name for name in all_columns if name != learn.dls.y_names[0]]

    # Check if embedding layers or other preprocessing steps affect the input size
    cat_names = learn.dls.cat_names
    cont_names = learn.dls.cont_names

    # Calculate the total input size to the first linear layer
    emb_szs = {name: learn.model.embeds[i].embedding_dim for i, name in enumerate(cat_names)}
    total_input_size = sum(emb_szs.values()) + len(cont_names)

    print(f"Embedding sizes: {emb_szs}")
    print(f"Total input size to the first linear layer: {total_input_size}")
    print(f"Shape of weights: {weights.shape}")

    # Ensure the number of weights matches the total input size
    if len(weights) != total_input_size:
        raise ValueError(f"Number of weights ({len(weights)}) does not match total input size ({total_input_size}).")

    # Assign importance to each feature
    idx = 0
    for name in cat_names:
        emb_size = emb_szs[name]
        importance[name] = weights[idx:idx+emb_size].mean().item()  # Average the importance across the embedding dimensions
        idx += emb_size
    for name in cont_names:
        importance[name] = weights[idx].item()
        idx += 1
    
    return importance


def chart_feature_importance(learn):
    # Calculate and display feature importance
    importance = get_feature_importance(learn)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nFeature Importances {len(importance)}:")
    for name, imp in sorted_importance:
        print_to_log_info(f"{name}: {imp:.4f}")

    # Visualize the importance
    from matplotlib import pyplot as plt

    plt.figure(figsize=(24, 4))
    plt.bar(range(len(importance)), [imp for name, imp in sorted_importance])
    plt.xticks(range(len(importance)), [name for name, imp in sorted_importance], rotation=45, ha='right')
    plt.title('Feature Importance')
    #plt.tight_layout()
    plt.show()


# todo: Single_Dummy_Features?
# todo: opponent features: Opponents_HCP 
# assert df_pretrained.columns[df_pretrained.isna().any()].empty, df_pretrained.columns[df_pretrained.isna().any()]

#df_pretrained = df_pretrained.astype({col:dtype for regex_col, dtype in astype_cols.items() for col in df_pretrained.filter(regex=regex_col)})


# FillMissing is causing following error. It is a known issue in fastai.
# c:\Users\bsali\miniconda3\envs\bridge12\Lib\site-packages\fastai\tabular\core.py:312: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

def train_model(df, y_names, cat_names=None, cont_names=None, nsamples=None, procs=[Categorify, FillMissing, Normalize], valid_pct=0.2, bs=1024*10, layers=None, epochs=3, device='cpu',y_range=(0,1)):

    # todo: disallow String/Utf8 and force use of pl.Categorical?

    # setup columns and validate
    print(f"{y_names=}")
    assert isinstance(y_names,list) and len(y_names) == 1, 'Only one target variable is supported.'

    print(df.describe())
    unimplemented_dtypes = df.select(pl.exclude(pl.Boolean,pl.Categorical,pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.Float32,pl.Float64,pl.String,pl.UInt8,pl.Utf8)).columns
    print(f"{unimplemented_dtypes=}") # todo: how to deal with these?

    # setup cat and cont names. All columns are assumed to be either cat or cont. Booleans should be in cat. Ints could be in either.
    if cat_names is None:
        cat_names = list(set(df.select(pl.col([pl.Boolean,pl.Categorical,pl.String])).columns).difference(y_names))
    print(f"{cat_names=}")
    if cont_names is None:
        cont_names = list(set(df.columns).difference(cat_names + y_names)) # pl.Datetime,pl.Float32,pl.Float64,pl.Int32,pl.Int64
    print(f"{cont_names=}")
    assert set(y_names).intersection(cat_names+cont_names) == set(), set(y_names).intersection(cat_names+cont_names)
    assert set(cat_names).intersection(cont_names) == set(), set(cat_names).intersection(cont_names)

    if nsamples is None:
        pandas_df = df[y_names+cat_names+cont_names].to_pandas()
    else:
        pandas_df = df[y_names+cat_names+cont_names].sample(nsamples,seed=42).to_pandas()

    print('y_names[0].dtype:',pandas_df[y_names[0]].dtype.name)
    if pandas_df[y_names[0]].dtype.name in ['boolean','category','object','string','uint8']: # 'object' is a probably a string. todo: What about ints? should classifier or regression be called?
        learn = train_classifier(pandas_df, y_names, cat_names, cont_names, procs=procs, valid_pct=min(valid_pct,10000/len(pandas_df)), bs=bs, layers=layers, epochs=epochs, device=device)
    elif pandas_df[y_names[0]].dtype.name in ['float32','float64']:
        learn = train_regression(pandas_df, y_names, cat_names, cont_names, procs=procs, valid_pct=min(valid_pct,10000/len(pandas_df)), bs=bs, layers=layers, epochs=epochs, device=device, y_range=y_range)
    else:
        raise ValueError(f"y_names dtype of {pandas_df[y_names[0]].dtype.name} not supported.")

    return learn
