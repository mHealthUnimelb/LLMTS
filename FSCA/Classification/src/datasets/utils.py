"""
Code to load Time Series Regression datasets. From:
https://github.com/ChangWeiTan/TSRegression/blob/master/utils
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import random
from sklearn import model_selection

import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from datasets.person_activity import PersonActivity

regression_datasets = ["AustraliaRainfall",
                       "HouseholdPowerConsumption1",
                       "HouseholdPowerConsumption2",
                       "BeijingPM25Quality",
                       "BeijingPM10Quality",
                       "Covid3Month",
                       "LiveFuelMoistureContent",
                       "FloodModeling1",
                       "FloodModeling2",
                       "FloodModeling3",
                       "AppliancesEnergy",
                       "BenzeneConcentration",
                       "NewsHeadlineSentiment",
                       "NewsTitleSentiment",
                       "BIDMC32RR",
                       "BIDMC32HR",
                       "BIDMC32SpO2",
                       "IEEEPPG",
                       "PPGDalia"]


def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


# The following code is adapted from the python package sktime to read .ts file.
class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """
    pass


def load_from_tsfile_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True,
                                  replace_missing_vals_with='NaN'):
    """Loads data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
    """

    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_target_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_float = None
    previous_timestamp_was_int = None
    previous_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    target_labels = False

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            # print(".", end='')
            # Strip white space from start/end of line and change to lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("problemname tag requires an associated value")

                    problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException("timestamps tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise TsFileParseException("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)
                    if token_len != 2:
                        raise TsFileParseException("univariate tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        univariate = True
                    elif tokens[1] == "false":
                        univariate = False
                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("classlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise TsFileParseException("if the classlabel tag is true then class values must be supplied")

                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("targetlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        target_labels = True
                    elif tokens[1] == "false":
                        target_labels = False
                    else:
                        raise TsFileParseException("invalid targetLabel value")

                    has_target_labels_tag = True
                    class_val_list = []
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise TsFileParseException("data tag should not have an associated value")

                    if data_started and not metadata_started:
                        raise TsFileParseException("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                    incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                    if incomplete_regression_meta_data and incomplete_classification_meta_data:
                        raise TsFileParseException("a full set of metadata has not been provided before the data")

                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False

                        timestamps_for_dimension = []
                        values_for_dimension = []

                        this_line_num_dimensions = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(pd.Series())
                                    this_line_num_dimensions += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamps_for_dimension = []
                                    values_for_dimension = []

                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and target_labels:
                                        class_val = line[char_num:].strip()

                                        # if class_val not in class_val_list:
                                        #     raise TsFileParseException(
                                        #         "the class value '" + class_val + "' on line " + str(
                                        #             line_num + 1) + " is not valid")

                                        class_val_list.append(float(class_val))
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within the next tuple

                                        if line[char_num] != "(" and not target_labels:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not start with a '('")

                                        char_num += 1
                                        tuple_data = ""

                                        while char_num < line_len and line[char_num] != ")":
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if char_num >= line_len or line[char_num] != ")":
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not end with a ')'")

                                        # Read in any spaces immediately after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(line[char_num]):
                                            char_num += 1

                                        # Check if there is another value or dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                        last_comma_index = tuple_data.rfind(',')

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has no comma inside of it")

                                        try:
                                            value = tuple_data[last_comma_index + 1:]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                        # Check the type of timestamp that we have

                                        timestamp = tuple_data[0: last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = float(timestamp)
                                                timestamp_is_float = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_float = False

                                        if not timestamp_is_int and not timestamp_is_float:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                        if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                        if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        # Store the values

                                        timestamps_for_dimension += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then we store the type of timestamp we had

                                        if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                            previous_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_int is None and timestamp_is_int:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_float is None and timestamp_is_float:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = True

                                        # See if we should add the data for this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (this_line_num_dimensions + 1):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamps_for_dimension = pd.DatetimeIndex(timestamps_for_dimension)

                                            instance_list[this_line_num_dimensions].append(
                                                pd.Series(index=timestamps_for_dimension, data=values_for_dimension))
                                            this_line_num_dimensions += 1

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(pd.Series(dtype=np.float32))
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dimensions

                                if num_dimensions != this_line_num_dimensions:
                                    raise TsFileParseException("line " + str(
                                        line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check that we are not expecting some more data, and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ',' that is not followed by another tuple")

                        elif has_another_dimension and target_labels:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ':' while it should list a class value")

                        elif has_another_dimension and not target_labels:
                            if len(instance_list) < (this_line_num_dimensions + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dimensions].append(pd.Series())
                            this_line_num_dimensions += 1
                            num_dimensions = this_line_num_dimensions

                        # If this is the 1st line of data we have seen then note the dimensions

                        if not has_another_value and num_dimensions != this_line_num_dimensions:
                            raise TsFileParseException("line " + str(
                                line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check if we should have class values, and if so that they are contained in those listed in the metadata

                        if target_labels and len(class_val_list) == 0:
                            raise TsFileParseException("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if target_labels:
                                num_dimensions -= 1

                            for dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False

                        # See how many dimensions that the case whose data in represented in this line has
                        this_line_num_dimensions = len(dimensions)

                        if target_labels:
                            this_line_num_dimensions -= 1

                        # All dimensions should be included for all series, even if they are empty
                        if this_line_num_dimensions != num_dimensions:
                            raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                num_dimensions) + " but have read " + str(this_line_num_dimensions))

                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series())

                        if target_labels:
                            class_val_list.append(float(dimensions[num_dimensions].strip()))

            line_num += 1

    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
        complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

        if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
            raise TsFileParseException("metadata incomplete")
        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")
        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data['dim_' + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if target_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data['class_vals'] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise TsFileParseException("empty file")


def process_data(X, min_len, normalise=None):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param min_len:
    :param normalise:
    :return:
    """
    tmp = []
    for i in tqdm(range(len(X))):
        _x = X.iloc[i, :].copy(deep=True)

        # 1. find the maximum length of each dimension
        all_len = [len(y) for y in _x]
        max_len = max(all_len)

        # 2. adjust the length of each dimension
        _y = []
        for y in _x:
            # 2.1 fill missing values
            if y.isnull().any():
                y = y.interpolate(method='linear', limit_direction='both')

            # 2.2. if length of each dimension is different, uniformly scale the shorter ones to the max length
            if len(y) < max_len:
                y = uniform_scaling(y, max_len)
            _y.append(y)
        _y = np.array(np.transpose(_y))

        # 3. adjust the length of the series, chop of the longer series
        _y = _y[:min_len, :]

        # 4. normalise the series
        if normalise == "standard":
            scaler = StandardScaler().fit(_y)
            _y = scaler.transform(_y)
        if normalise == "minmax":
            scaler = MinMaxScaler().fit(_y)
            _y = scaler.transform(_y)

        tmp.append(_y)
    X = np.array(tmp)
    return X


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def preprocess_P12(PT_dict, arr_outcomes):
    """
    Process a list of patient records (PT_dict) and outcome values (arr_outcomes).
    Each patient record is assumed to have:
      - 'id': a record identifier (string)
      - 'static': a tuple of 5 static variables
      - 'arr': a numpy.ndarray of shape (T, 36) for T time steps
      - 'time': a numpy.ndarray of shape (T, 1) with time stamps (dynamic times)
      - 'length': the number T of valid time steps in arr and time

    The output for each patient is a tuple:
      (record_id, tt, vals, mask, outcome)
    where:
      - tt is a 1D tensor of time stamps. A new initial time stamp 0 is prepended for the static row.
      - vals is a 2D tensor of shape ((length + 1), 41) where the first row is the static data
        (5 static variables and 36 zeros) and the remaining rows are from 'arr' padded on the left
        with 5 zeros (static variables).
      - mask is built in a similar fashion: for the static row, the mask is 1 for the first 5 entries
        and 0 for the remaining 36; for dynamic rows, we compute the nonzero indicator from 'arr' and
        pad with 5 zeros on the left (static variables).
      - outcome is a tensor converted from arr_outcomes.
    """
    total = []
    for i, patient in enumerate(PT_dict):
        length = patient['length']
        record_id = patient['id']

        # # process static features (time = 0)
        # static_features = torch.tensor(patient['static'], dtype=torch.float32)  # shape: (5,)
        # static_row = torch.cat([static_features, torch.zeros(36, dtype=torch.float32)])  # shape: (41,)
        #
        # # For dynamic features, get the measurement array and pad with 5 zeros at the beginning.
        # arr_dynamic = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)  # shape: (length, 36)
        # dynamic_vals = torch.cat([torch.zeros((length, 5), dtype=torch.float32), arr_dynamic],
        #                          dim=1)  # shape: (length, 41)
        #
        # # concatenate static and dynamic features
        # vals = torch.cat([static_row.unsqueeze(0), dynamic_vals], dim=0)  # shape: (length+1, 41)

        # prepare the values array of shape [length+1, 5 (static) + 36 (dynamic) = 41]
        vals = torch.zeros((length + 1, 41), dtype=torch.float32)

        # fill row 0 (time = 0) with static variables in columns 0..4
        static_vars = torch.tensor(patient['static'], dtype=torch.float32)  # shape [5]
        vals[0, :5] = static_vars

        # fill rows [1..length] in columns [5..40] with the dynamic features
        dynamic_vars = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)  # shape [length, 36]
        vals[1:, 5:] = dynamic_vars

        # tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        dynamic_tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        tt = torch.zeros(length + 1, dtype=torch.float32)
        tt[1:] = dynamic_tt
        # vals = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)

        # # dynamic features
        # m = np.zeros(shape=patient['arr'][:length, :].shape)
        # m[patient['arr'][:length, :].nonzero()] = 1
        # dynamic_mask = torch.tensor(m, dtype=torch.float32)  # shape: (length, 36)
        # dynamic_mask = torch.cat([torch.zeros((length, 5), dtype=torch.float32), dynamic_mask],
        #                          dim=1)  # shape: (length, 41)
        # # static mask
        # static_mask = torch.cat([torch.ones(5, dtype=torch.float32), torch.zeros(36, dtype=torch.float32)])
        # mask = torch.cat([static_mask.unsqueeze(0), dynamic_mask], dim=0)  # shape: (length+1, 41)

        # mask
        mask = torch.zeros((length + 1, 41), dtype=torch.float32)

        # row 0, columns 0..4 are the static variables (mark these as present)
        mask[0, :5] = 1.0

        # for the time-series portion, copy the nonzero positions
        arr_np = patient['arr'][:length, :]  # shape [length, 36]
        mask_np = np.zeros_like(arr_np)
        mask_np[arr_np.nonzero()] = 1
        # put mask_np into columns [5..40] of rows [1..length]
        mask[1:, 5:] = torch.tensor(mask_np, dtype=torch.float32)

        # mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][-1], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total

# def preprocess_P12(PT_dict, arr_outcomes):
#     total = []
#     for i, patient in enumerate(PT_dict):
#         length = patient['length']
#         record_id = patient['id']
#         tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
#         vals = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)
#         m = np.zeros(shape=patient['arr'][:length, :].shape)
#         m[patient['arr'][:length, :].nonzero()] = 1
#         mask = torch.tensor(m, dtype=torch.float32)
#         outcome = torch.tensor(arr_outcomes[i][-1], dtype=torch.float32)
#         total.append((record_id, tt, vals, mask, outcome))
#
#     return total


def preprocess_P19(PT_dict, arr_outcomes, labels_ts):
    total = []
    for i, patient in enumerate(PT_dict):
        length = patient['length']
        record_id = patient['id']
        tt = torch.squeeze(torch.tensor(patient['time'][:length]), 1)
        vals = torch.tensor(patient['arr'][:length, :], dtype=torch.float32)
        m = np.zeros(shape=patient['arr'][:length, :].shape)
        m[patient['arr'][:length, :].nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][0], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total


def preprocess_eICU(PT_dict, arr_outcomes, labels_ts):
    total = []
    for i, patient in enumerate(PT_dict):
        record_id = str(i)
        tt = torch.squeeze(torch.tensor(patient['time']), 1)
        vals = torch.tensor(patient['arr'], dtype=torch.float32)
        m = np.zeros(shape=patient['arr'].shape)
        m[patient['arr'].nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))

    return total


def preprocess_PAM(PT_dict, arr_outcomes):
    length = 600
    total = []
    for i, patient in enumerate(PT_dict):
        record_id = str(i)
        tt = torch.tensor(list(range(length)))
        vals = torch.tensor(patient, dtype=torch.float32)
        m = np.zeros(shape=patient.shape)
        m[patient.nonzero()] = 1
        mask = torch.tensor(m, dtype=torch.float32)
        outcome = torch.tensor(arr_outcomes[i][0], dtype=torch.float32)
        total.append((record_id, tt, vals, mask, outcome))
    return total


def random_sample(idx_0, idx_1, batch_size):
    """
    Returns a balanced sample by randomly sampling without replacement.

    :param idx_0: indices of negative samples
    :param idx_1: indices of positive samples
    :param batch_size: batch size
    :return: indices of balanced batch of negative and positive samples
    """
    idx0_batch = np.random.choice(idx_0, size=int(batch_size / 2), replace=False)
    idx1_batch = np.random.choice(idx_1, size=int(batch_size / 2), replace=False)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def random_sample_8(ytrain, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(np.where(ytrain == 0)[0], size=int(B / 8), replace=replace)
    idx1_batch = np.random.choice(np.where(ytrain == 1)[0], size=int(B / 8), replace=replace)
    idx2_batch = np.random.choice(np.where(ytrain == 2)[0], size=int(B / 8), replace=replace)
    idx3_batch = np.random.choice(np.where(ytrain == 3)[0], size=int(B / 8), replace=replace)
    idx4_batch = np.random.choice(np.where(ytrain == 4)[0], size=int(B / 8), replace=replace)
    idx5_batch = np.random.choice(np.where(ytrain == 5)[0], size=int(B / 8), replace=replace)
    idx6_batch = np.random.choice(np.where(ytrain == 6)[0], size=int(B / 8), replace=replace)
    idx7_batch = np.random.choice(np.where(ytrain == 7)[0], size=int(B / 8), replace=replace)
    idx = np.concatenate(
        [idx0_batch, idx1_batch, idx2_batch, idx3_batch, idx4_batch, idx5_batch, idx6_batch, idx7_batch], axis=0)
    return idx


def balanced_batch_sampler(train_data, true_labels, batch_size, n_classes):
    """
        Creates an upsampled training dataset with balanced batches.

        Each batch contains an equal number of samples from each class.
        Samples are drawn randomly without immediate repetition. When the
        available pool for a class is exhausted, it is refilled and reshuffled.

        Args:
            train_data (list or array): List of training samples.
            true_labels (np.array): Array of labels corresponding to train_data.
            batch_size (int): Total batch size; must be divisible by n_classes.
            n_classes (int): Number of classes.

        Returns:
            list: Upsampled training data with balanced batches.
        """
    # Ensure batch_size is divisible by the number of classes
    if batch_size % n_classes != 0:
        raise ValueError("batch_size must be divisible by n_classes")

    # Number of samples per class per batch
    per_class_per_batch = batch_size // n_classes

    # Create a dictionary for the full list of indices for each class
    class_indices = {}
    # Also maintain an available pool for each class from which samples are drawn
    available_indices = {}
    for cls in range(n_classes):
        indices = np.where(true_labels == cls)[0].tolist()
        class_indices[cls] = indices
        available_indices[cls] = indices.copy()
        # np.random.shuffle(available_indices[cls])

    # Decide on the number of iterations of batches to generate in this epoch.
    # Here, we ensure the total upsampled data covers at least the size of the original dataset.
    num_iter_batch = int(np.ceil(len(true_labels) / batch_size))

    upsampled_train_data = []

    for _ in range(num_iter_batch):
        batch_indices = []
        for cls in range(n_classes):
            sampled = []
            # Use leftover samples first if available.
            num_available = len(available_indices[cls])
            if num_available >= per_class_per_batch:
                # Enough available samples: take the first 'per_class_per_batch' samples.
                sampled = available_indices[cls][:per_class_per_batch]
                available_indices[cls] = available_indices[cls][per_class_per_batch:]
            else:
                # Not enough samples remaining: use all the available samples.
                if num_available > 0:
                    sampled = available_indices[cls].copy()
                    available_indices[cls] = []
                # Calculate how many additional samples are needed.
                needed = per_class_per_batch - len(sampled)
                # Refill the pool by shuffling a complete copy of the class indices.
                new_pool = class_indices[cls].copy()
                np.random.shuffle(new_pool)
                additional_samples = new_pool[:needed]
                sampled.extend(additional_samples)
                # Store the remaining samples in the new pool for future use.
                available_indices[cls] = new_pool[needed:]
            batch_indices.extend(sampled)

        # Optionally shuffle the combined batch indices for randomness within the batch
        # np.random.shuffle(batch_indices)
        # Append the samples corresponding to these indices to the upsampled training data
        for idx in batch_indices:
            upsampled_train_data.append(train_data[idx])

        print("batch_indices: ", batch_indices)

    return upsampled_train_data


def get_data(args, dataset, device, q=0.016, upsampling_batch=True, flag=1):
    print("upsampling_batch", upsampling_batch)
    print("args.classif", args['classif'])
    if dataset == 'P12':
        total_dataset = PhysioNet('datasets/physionet',
                                  quantization=q,
                                  download=True,
                                  device=device)
        PT_dict = np.load('./datasets/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
        # arr_outcomes = np.load('./datasets/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)

        idx_train, idx_val, idx_test = np.load(args['data_split_path'], allow_pickle=True)
    elif dataset == 'P19':
        PT_dict = np.load('../P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        labels_ts = np.load('../P19data/processed_data/labels_ts.npy', allow_pickle=True)
        labels_demogr = np.load('../P19data/processed_data/labels_demogr.npy', allow_pickle=True)
        arr_outcomes = np.load('../P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)

        total_dataset = preprocess_P19(PT_dict, arr_outcomes, labels_ts)
    elif dataset == 'eICU':
        PT_dict = np.load('../../../eICUdata/processed_data/PTdict_list.npy', allow_pickle=True)
        labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
        labels_demogr = np.load('../../../eICUdata/processed_data/eICU_static_vars.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../eICUdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_eICU(PT_dict, arr_outcomes, labels_ts)

    elif dataset == 'PAM':
        PT_dict = np.load('./data/PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('./data/PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_PAM(PT_dict, arr_outcomes)

    elif dataset == 'MIMIC':
        total_dataset = torch.load('./datasets/MIMIC/mimic_classification/processed/mimic.pt', map_location='cpu')
        total_dataset = [(record_id, tt, vals, mask, torch.tensor(label, dtype=torch.long)) for
                         (record_id, tt, vals, mask, label) in total_dataset]

    elif dataset == 'activity':
        # args.pred_window = 1000
        total_dataset = PersonActivity('datasets/activity/', n_samples = int(1e8), download=True, device = device)
        # total_dataset = torch.load('./data/activiaty/processed/data.pt', map_location='cpu')


    print('len(total_dataset):', len(total_dataset))
    print("total_dataset[0]:", total_dataset[0])

    global_tt = torch.unique(torch.cat([tpl[1] for tpl in total_dataset]), sorted=True)

    # if split_type == 'random':
    #     # Shuffle and split
    #     train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.9,
    #                                                              shuffle=True)  # 80% train, 10% validation, 10% test

    if dataset == 'P12':
        # get recorde_id from PTdict_list.npy
        print("idx_train[0]", idx_train[0])
        train_record_ids = [PT_dict[i]['id'] for i in idx_train]
        print("train_record_ids[0]", train_record_ids[0])
        val_record_ids = [PT_dict[i]['id'] for i in idx_val]
        test_record_ids = [PT_dict[i]['id'] for i in idx_test]

        #  dictionary mapping record_id to its tuple
        record_dict = {rec[0]: rec for rec in total_dataset}

        # get train/val/test data
        train_data = [record_dict[rid] for rid in train_record_ids]
        val_data = [record_dict[rid] for rid in val_record_ids]
        test_data = [record_dict[rid] for rid in test_record_ids]

        print("train_data[0]:", train_data[0])
        print("val_data[0]:", val_data[0])
        print("test_data[0]:", test_data[0])
    elif dataset == 'MIMIC' or dataset == 'activity':
        print("seed", args['seed'])
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=args['seed'],
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=args['seed'],
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
    else:
        train_data = [total_dataset[i] for i in idx_train]
        print("train_data[0]:", train_data[0])
        val_data = [total_dataset[i] for i in idx_val]
        print("val_data[0]:", val_data[0])
        test_data = [total_dataset[i] for i in idx_test]
        print("test_data[0]:", test_data[0])

    # elif split_type == 'age' or split_type == 'gender':
    #     if dataset == 'P12':
    #         prefix = 'mtand'
    #     elif dataset == 'P19':
    #         prefix = 'P19'
    #     elif dataset == 'eICU':   # possible only with split_type == 'gender'
    #         prefix = 'eICU'
    #
    #     if split_type == 'age':
    #         if dataset == 'eICU':
    #             print('\nCombination of eICU dataset and age split is not possible.\n')
    #         if reverse == False:
    #             idx_train = np.load('%s_idx_under_65.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_over_65.npy' % prefix, allow_pickle=True)
    #         else:
    #             idx_train = np.load('%s_idx_over_65.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_under_65.npy' % prefix, allow_pickle=True)
    #     elif split_type == 'gender':
    #         if reverse == False:
    #             idx_train = np.load('%s_idx_male.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_female.npy' % prefix, allow_pickle=True)
    #         else:
    #             idx_train = np.load('%s_idx_female.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_male.npy' % prefix, allow_pickle=True)
    #
    #     np.random.shuffle(idx_train)
    #     np.random.shuffle(idx_vt)
    #     train_data = [total_dataset[i] for i in idx_train]
    #     test_data = [total_dataset[i] for i in idx_vt]

    # tt: time steps, vals: observed values, mask: which values are observed
    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)  # determine the number of features. vals: [T, D], where D is the number of features
    data_min, data_max = get_data_min_max(total_dataset,
                                          device)  # Compute the minimum and maximum values across all features in the entire dataset
    # batch_size = 128
    batch_size = min(len(train_data),
                     args['batch_size'])  # ensures the batch size isn't larger than the dataset or user-specified number

    if flag:
        if args['classif']:
            # if split_type == 'random':
            #     train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8889,
            #                                                             shuffle=False)  # 80% train, 10% validation, 10% test
            print("train len:", len(train_data))
            print("val len:", len(val_data))
            print("test len:", len(test_data))
            # elif split_type == 'age' or split_type == 'gender':
            #     val_data, test_data = model_selection.train_test_split(test_data, train_size=0.5, shuffle=False)

            # if dataset == 'P12':
            #     num_all_features = 36
            # elif dataset == 'P19':
            #     num_all_features = 34
            # elif dataset == 'eICU':
            #     num_all_features = 14
            # elif dataset == 'PAM':
            #     num_all_features = 17

            # num_missing_features = round(missing_ratio * num_all_features)
            # if feature_removal_level == 'sample':
            #     for i, tpl in enumerate(val_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)
            # elif feature_removal_level == 'set':
            #     if dataset == 'P12':
            #         dict_params = total_dataset.params_dict
            #         density_scores_names = np.load('../saved/IG_density_scores_P12.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'P19':
            #         labels_ts = np.load('../../../P19data/processed_data/labels_ts.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts[:-1])}
            #         density_scores_names = np.load('../saved/IG_density_scores_P19.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'eICU':
            #         labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts)}
            #         density_scores_names = np.load('../saved/IG_density_scores_eICU.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'PAM':
            #         density_scores_indices = np.load('../saved/IG_density_scores_PAM.npy', allow_pickle=True)[:, 0]
            #         idx = list(map(int, density_scores_indices[:num_missing_features]))
            #
            #     for i, tpl in enumerate(val_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)

            if upsampling_batch:
                train_data_upsamled = []
                true_labels = np.array([float(x[4].item()) for x in train_data])
                # true_labels = np.array(list(map(lambda x: float(x[7]), np.array(train_data)[:, 4])))
                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':  # 2 classes
                    idx_0 = np.where(true_labels == 0)[0]
                    print("idx_0 length", len(idx_0))
                    idx_1 = np.where(true_labels == 1)[0]
                    print("idx_1 length", len(idx_1))
                    # Method 1
                    # for _ in range(len(true_labels) // batch_size):
                    #     indices = random_sample(idx_0, idx_1, batch_size)
                    #     for i in indices:
                    #         train_data_upsamled.append(train_data[i])

                    # Method 2
                    train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 2)

                elif dataset == 'PAM':  # 8 classes
                    # for b in range(len(true_labels) // batch_size):
                    #     indices = random_sample_8(true_labels, batch_size)
                    #     for i in indices:
                    #         train_data_upsamled.append(train_data[i])
                    train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 8)

                train_data = train_data_upsamled

            if dataset == 'activity':
                test_data_combined = variable_time_collate_fn_activity(test_data, args, device, classify=args['classif'], activity=True)
                train_data_combined = variable_time_collate_fn_activity(train_data, args, device, classify=args['classif'], activity=True)
                val_data_combined = variable_time_collate_fn_activity(val_data, args, device, classify=args['classif'], activity=True)
            else:
                test_data_combined = variable_time_collate_fn(test_data, args, device, classify=args['classif'], data_min=data_min,
                                                            data_max=data_max, global_tt=global_tt)
                train_data_combined = variable_time_collate_fn(train_data, args, device, classify=args['classif'], data_min=data_min,
                                                            data_max=data_max, global_tt=global_tt)
                val_data_combined = variable_time_collate_fn(
                    val_data, args, device, classify=args['classif'], data_min=data_min, data_max=data_max, global_tt=global_tt)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            # convert the combined data (a tuple of data and labels) into TensorDatasets
            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            # if not classification (e.g., regression/forecasting)
            train_data_combined = variable_time_collate_fn(
                train_data, args, device, classify=args['classif'], data_min=data_min, data_max=data_max)

        # shuffle=False since it's handled above
        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        # if flag is not set, use variable_time_collate_fn2 for custom handling
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                         data_type="train",
                                                                                         data_min=data_min,
                                                                                         data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                        data_type="test",
                                                                                        data_min=data_min,
                                                                                        data_max=data_max))

    data_objects = {"dataset_obj": {},
                    "train_data": train_data,
                    "train_dataloader": train_dataloader,
                    "test_data": test_data,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,  # number of features
                    "n_train_batches": len(train_dataloader),  # number of batches in train
                    "n_test_batches": len(test_dataloader),
                    "attr": {},  # optional
                    "classif_per_tp": False,  # (optional) boolean flag indicating classification per time point or not
                    "n_labels": 1}  # (optional) how many labels per sample are expected
    if args['classif']:
        # if classification, also create and store a validation DataLoader
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_data"] = val_data
        data_objects["val_dataloader"] = val_dataloader
    return data_objects  # return all the prepared data and metadata as a dictionary

def get_data_mTAN(args, dataset, device, q, upsampling_batch, flag=1):
    print("upsampling_batch", upsampling_batch)
    # print("split_type", split_type)
    if dataset == 'P12':
        total_dataset = PhysioNet('datasets/physionet',
                                  quantization=q,
                                  download=True,
                                  device=device)
        PT_dict = np.load('./datasets/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('./datasets/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)
        idx_train, idx_val, idx_test = np.load(args['data_split_path'], allow_pickle=True)

        # total_dataset = preprocess_P12(PT_dict, arr_outcomes)
    elif dataset == 'P19':
        PT_dict = np.load('../P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        labels_ts = np.load('../P19data/processed_data/labels_ts.npy', allow_pickle=True)
        labels_demogr = np.load('../P19data/processed_data/labels_demogr.npy', allow_pickle=True)
        arr_outcomes = np.load('../P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)

        total_dataset = preprocess_P19(PT_dict, arr_outcomes, labels_ts)
    elif dataset == 'eICU':
        PT_dict = np.load('../../../eICUdata/processed_data/PTdict_list.npy', allow_pickle=True)
        labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
        labels_demogr = np.load('../../../eICUdata/processed_data/eICU_static_vars.npy', allow_pickle=True)
        arr_outcomes = np.load('../../../eICUdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_eICU(PT_dict, arr_outcomes, labels_ts)

    elif dataset == 'PAM':
        # print("current path", os.getcwd())
        PT_dict = np.load('./data/PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load('./data/PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)

        total_dataset = preprocess_PAM(PT_dict, arr_outcomes)

    elif dataset == 'MIMIC':
        total_dataset = torch.load('./datasets/MIMIC/mimic_classification/processed/mimic.pt', map_location='cpu')
        total_dataset = [(record_id, tt, vals, mask, torch.tensor(label)) for
                         (record_id, tt, vals, mask, label) in total_dataset]

    elif dataset == 'activity':
        # args.pred_window = 1000
        total_dataset = PersonActivity('datasets/activity/', n_samples = int(1e8), download=True, device = device)
        # total_dataset = torch.load('./data/activiaty/processed/data.pt', map_location='cpu')


    print('len(total_dataset):', len(total_dataset))

    global_tt = torch.unique(torch.cat([tpl[1] for tpl in total_dataset]), sorted=True)

    # if split_type == 'random':
    #     # Shuffle and split
    #     train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.9,
    #                                                              shuffle=True)  # 80% train, 10% validation, 10% test

    if dataset == 'P12':
        # get recorde_id from PTdict_list.npy
        print("idx_train[0]", idx_train[0])
        train_record_ids = [PT_dict[i]['id'] for i in idx_train]
        print("train_record_ids[0]", train_record_ids[0])
        val_record_ids = [PT_dict[i]['id'] for i in idx_val]
        test_record_ids = [PT_dict[i]['id'] for i in idx_test]

        #  dictionary mapping record_id to its tuple
        record_dict = {rec[0]: rec for rec in total_dataset}

        # get train/val/test data
        train_data = [record_dict[rid] for rid in train_record_ids]
        val_data = [record_dict[rid] for rid in val_record_ids]
        test_data = [record_dict[rid] for rid in test_record_ids]

        print("train_data[0]:", train_data[0])
        print("val_data[0]:", val_data[0])
        print("test_data[0]:", test_data[0])
    elif dataset == 'MIMIC' or dataset == 'activity':
        print("seed", args['seed'])
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=args['seed'],
                                                                shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=args['seed'],
                                                                shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
    else:
        train_data = [total_dataset[i] for i in idx_train]
        print("train_data[0]:", train_data[0])
        val_data = [total_dataset[i] for i in idx_val]
        print("val_data[0]:", val_data[0])
        test_data = [total_dataset[i] for i in idx_test]
        print("test_data[0]:", test_data[0])

    # y_train = np.array([x[-1].item() for x in train_data])
    # y_val = np.array([x[-1].item() for x in val_data])
    # y_test = np.array([x[-1].item() for x in test_data])
    #
    # # compute class weights
    # w_train = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # w_val = compute_class_weight('balanced', classes=np.unique(y_val), y=y_val)
    # w_test = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)

    # train_data = []
    # val_data = []
    # test_data = []

    # for i in total_dataset:
    #     if i[0] in idx_train:
    #         train_data.append(i)
    #     elif i[0] in idx_val:
    #         val_data.append(i)
    #     elif i[0] in idx_test:
    #         test_data.append(i)

    # elif split_type == 'age' or split_type == 'gender':
    #     if dataset == 'P12':
    #         prefix = 'mtand'
    #     elif dataset == 'P19':
    #         prefix = 'P19'
    #     elif dataset == 'eICU':   # possible only with split_type == 'gender'
    #         prefix = 'eICU'
    #
    #     if split_type == 'age':
    #         if dataset == 'eICU':
    #             print('\nCombination of eICU dataset and age split is not possible.\n')
    #         if reverse == False:
    #             idx_train = np.load('%s_idx_under_65.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_over_65.npy' % prefix, allow_pickle=True)
    #         else:
    #             idx_train = np.load('%s_idx_over_65.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_under_65.npy' % prefix, allow_pickle=True)
    #     elif split_type == 'gender':
    #         if reverse == False:
    #             idx_train = np.load('%s_idx_male.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_female.npy' % prefix, allow_pickle=True)
    #         else:
    #             idx_train = np.load('%s_idx_female.npy' % prefix, allow_pickle=True)
    #             idx_vt = np.load('%s_idx_male.npy' % prefix, allow_pickle=True)
    #
    #     np.random.shuffle(idx_train)
    #     np.random.shuffle(idx_vt)
    #     train_data = [total_dataset[i] for i in idx_train]
    #     test_data = [total_dataset[i] for i in idx_vt]

    # tt: time steps, vals: observed values, mask: which values are observed
    record_id, tt, vals, mask, labels = train_data[0]

    input_dim = vals.size(-1)  # determine the number of features. vals: [T, D], where D is the number of features
    data_min, data_max = get_data_min_max(total_dataset,
                                          device)  # Compute the minimum and maximum values across all features in the entire dataset
    # batch_size = 128
    batch_size = min(len(train_data),
                     args['batch_size'])  # ensures the batch size isn't larger than the dataset or user-specified number

    if flag:
        if args['classif']:
            # if split_type == 'random':
            #     train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8889,
            #                                                             shuffle=False)  # 80% train, 10% validation, 10% test
            print("train len:", len(train_data))
            print("val len:", len(val_data))
            print("test len:", len(test_data))
            # elif split_type == 'age' or split_type == 'gender':
            #     val_data, test_data = model_selection.train_test_split(test_data, train_size=0.5, shuffle=False)

            # if dataset == 'P12':
            #     num_all_features = 36
            # elif dataset == 'P19':
            #     num_all_features = 34
            # elif dataset == 'eICU':
            #     num_all_features = 14
            # elif dataset == 'PAM':
            #     num_all_features = 17

            # num_missing_features = round(missing_ratio * num_all_features)
            # if feature_removal_level == 'sample':
            #     for i, tpl in enumerate(val_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)
            # elif feature_removal_level == 'set':
            #     if dataset == 'P12':
            #         dict_params = total_dataset.params_dict
            #         density_scores_names = np.load('../saved/IG_density_scores_P12.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'P19':
            #         labels_ts = np.load('../../../P19data/processed_data/labels_ts.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts[:-1])}
            #         density_scores_names = np.load('../saved/IG_density_scores_P19.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'eICU':
            #         labels_ts = np.load('../../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
            #         dict_params = {label: i for i, label in enumerate(labels_ts)}
            #         density_scores_names = np.load('../saved/IG_density_scores_eICU.npy', allow_pickle=True)[:, 1]
            #         idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
            #     elif dataset == 'PAM':
            #         density_scores_indices = np.load('../saved/IG_density_scores_PAM.npy', allow_pickle=True)[:, 0]
            #         idx = list(map(int, density_scores_indices[:num_missing_features]))
            #
            #     for i, tpl in enumerate(val_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         val_data[i] = tuple(tpl)
            #     for i, tpl in enumerate(test_data):
            #         _, _, values, _, _ = tpl
            #         tpl = list(tpl)
            #         values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
            #         tpl[2] = values
            #         test_data[i] = tuple(tpl)

            if upsampling_batch:
                train_data_upsamled = []
                true_labels = np.array([float(x[4].item()) for x in train_data])
                # true_labels = np.array(list(map(lambda x: float(x[7]), np.array(train_data)[:, 4])))
                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':  # 2 classes
                    idx_0 = np.where(true_labels == 0)[0]
                    print("idx_0 length", len(idx_0))
                    idx_1 = np.where(true_labels == 1)[0]
                    print("idx_1 length", len(idx_1))
                    # Method 1
                    # for _ in range(len(true_labels) // batch_size):
                    #     indices = random_sample(idx_0, idx_1, batch_size)
                    #     for i in indices:
                    #         train_data_upsamled.append(train_data[i])

                    # Method 2
                    train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 2)

                elif dataset == 'PAM':  # 8 classes
                    # for b in range(len(true_labels) // batch_size):
                    #     indices = random_sample_8(true_labels, batch_size)
                    #     for i in indices:
                    #         train_data_upsamled.append(train_data[i])
                    train_data_upsamled = balanced_batch_sampler(train_data, true_labels, batch_size, 8)

                train_data = train_data_upsamled

            if dataset == 'activity':
                # test_data_combined = variable_time_collate_fn_activity(test_data, args, device, classify=args.classif, activity=True)
                test_data_combined = variable_time_collate_fn_mTAN(test_data, args, device, classify=args['classif'], activity=True)
                # train_data_combined = variable_time_collate_fn_activity(train_data, args, device, classify=args.classif, activity=True)
                train_data_combined = variable_time_collate_fn_activity(train_data, args, device, classify=args['classif'], activity=True)
                # val_data_combined = variable_time_collate_fn_activity(val_data, args, device, classify=args.classif, activity=True)
                val_data_combined = variable_time_collate_fn_activity(val_data, args, device, classify=args['classif'], activity=True)
            else:
                test_data_combined = variable_time_collate_fn_mTAN(test_data, args, device, classify=args['classif'], data_min=data_min,
                                                            data_max=data_max)
                train_data_combined = variable_time_collate_fn_mTAN(train_data, args, device, classify=args['classif'], data_min=data_min,
                                                            data_max=data_max)
                val_data_combined = variable_time_collate_fn_mTAN(
                    val_data, args, device, classify=args['classif'], data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            # convert the combined data (a tuple of data and labels) into TensorDatasets
            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            # if not classification (e.g., regression/forecasting)
            train_data_combined = variable_time_collate_fn_mTAN(
                train_data, device, classify=args['classif'], data_min=data_min, data_max=data_max)

        # shuffle=False since it's handled above
        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        # if flag is not set, use variable_time_collate_fn2 for custom handling
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                         data_type="train",
                                                                                         data_min=data_min,
                                                                                         data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device,
                                                                                        data_type="test",
                                                                                        data_min=data_min,
                                                                                        data_max=data_max))

    data_objects = {"dataset_obj": {},
                    "train_data": train_data,
                    "train_dataloader": train_dataloader,
                    "test_data": test_data,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,  # number of features
                    "n_train_batches": len(train_dataloader),  # number of batches in train
                    "n_test_batches": len(test_dataloader),
                    "attr": {},  # optional
                    "classif_per_tp": False,  # (optional) boolean flag indicating classification per time point or not
                    "n_labels": 1}  # (optional) how many labels per sample are expected
    if args['classif']:
        # if classification, also create and store a validation DataLoader
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_data"] = val_data
        data_objects["val_dataloader"] = val_dataloader
    return data_objects  # return all the prepared data and metadata as a dictionary

def variable_time_collate_fn(batch, args, device=torch.device("cpu"), classify=False,
                             data_min=None, data_max=None, global_tt=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    # print("batch shape:", batch.shape)
    D = batch[0][2].shape[1]
    # combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    # combined_tt = combined_tt.to(device)
    combined_tt = global_tt.to(device)
    print("combined_tt shape", combined_tt.shape)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)
    # combined_tt = combined_tt.unsqueeze(0).expand(len(batch), -1)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # indices = inverse_indices[offset:offset + len(tt)]
        # offset += len(tt)
        indices = torch.searchsorted(combined_tt, tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels


    combined_vals, _, _ = normalize_masked_data(combined_vals, combined_mask, att_min=data_min, att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    B = combined_vals.size(0)
    T = combined_tt.size(0)
    combined_tt = combined_tt.view(1, T, 1).expand(B, T, 1).to(device)
    print("combined_tt shape", combined_tt.shape)
    combined_data = torch.cat((combined_vals, combined_mask, combined_tt), 2)

    if classify:
        return combined_data, combined_labels
    else:
        return combined_data

def variable_time_collate_fn_mTAN(batch, args, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                if labels is not None:
                    combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data
    

def variable_time_collate_fn_activity(batch, args, device=torch.device("cpu"), classify=False, activity=True, data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data