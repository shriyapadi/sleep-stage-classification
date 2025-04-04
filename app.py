import hashlib

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyedflib import highlevel
from librosa.feature import melspectrogram
from librosa import power_to_db
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Reshape,
    Flatten,
    TimeDistributed,
    Bidirectional,
    BatchNormalization,
    Dropout,
    Input,
    Add,
    Masking,
    Conv2D,
    MaxPooling2D,
    Concatenate,
    Activation
)
from tensorflow.keras.regularizers import l2
import io
import tempfile
import os
import warnings

warnings.filterwarnings("error")

# Constants
FS = 100  # Sampling rate
SPEC_LEN = 30  # Length of spectrogram in seconds
n_fft = 256
hop_length = 64
n_mels = 64
SPECTROGRAM_FREQS = 64
SPECTROGRAM_LEN = 47
NUM_OUTPUT_CLASSES = 3
CONV_OUTPUT_LEN = 30

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# User credentials (in production, use a secure database)
USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "full_name": "Administrator"
    },
    "user": {
        "password": hashlib.sha256("user123".encode()).hexdigest(),
        "full_name": "Shriya"
    }
}


def check_password(username: str, password: str) -> bool:
    """Verify username and password"""
    if username in USERS:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        return hashed_pw == USERS[username]["password"]
    return False


def login_page():
    """Display the login page"""
    st.title("Sleep Stage Interpretation using XAI")
    st.subheader("Login")

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Create a form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if check_password(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # Add some information about available users
        st.markdown("""
        ---
        **Test Credentials:**
        - Admin: username=`admin`, password=`admin123`
        - User: username=`user`, password=`user123`
        """)


def logout():
    """Log out the user"""
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.rerun()


def create_model():
    """Recreate the model architecture from train_model.py"""
    # Define shared layers
    shared_conv1 = Conv2D(filters=8, kernel_size=3, padding='same')
    shared_bn1 = BatchNormalization()
    shared_relu1 = Activation('relu')
    shared_mp1 = MaxPooling2D(pool_size=4, strides=4)
    shared_do1 = Dropout(0.2)

    shared_conv2 = Conv2D(filters=16, kernel_size=3, padding='same')
    shared_bn2 = BatchNormalization()
    shared_relu2 = Activation('relu')
    shared_mp2 = MaxPooling2D(pool_size=4, strides=4)
    shared_do2 = Dropout(0.2)

    shared_flatten = Flatten()
    shared_dense1 = Dense(units=CONV_OUTPUT_LEN)
    shared_bn4 = BatchNormalization()
    shared_relu4 = Activation('relu')

    def build_cnn_per_channel(input_tensor, ch):
        """Build CNN for each channel"""
        cnn_output = shared_conv1(input_tensor[:, :, :, ch][..., None])
        cnn_output = shared_bn1(cnn_output)
        cnn_output = shared_relu1(cnn_output)
        cnn_output = shared_mp1(cnn_output)
        cnn_output = shared_do1(cnn_output)
        cnn_output = shared_conv2(cnn_output)
        cnn_output = shared_bn2(cnn_output)
        cnn_output = shared_relu2(cnn_output)
        cnn_output = shared_mp2(cnn_output)
        cnn_output = shared_do2(cnn_output)
        cnn_output = shared_flatten(cnn_output)
        cnn_output = shared_dense1(cnn_output)
        cnn_output = shared_bn4(cnn_output)
        cnn_output = shared_relu4(cnn_output)
        cnn_output = Reshape((1, CONV_OUTPUT_LEN))(cnn_output)
        return cnn_output

    def build_cnn(input_tensor):
        """Build complete CNN with both channels"""
        cnn_output0 = build_cnn_per_channel(input_tensor, 0)
        cnn_output1 = build_cnn_per_channel(input_tensor, 1)
        cnn_output = Concatenate(axis=1)([cnn_output0, cnn_output1])
        cnn_output = Reshape((1, CONV_OUTPUT_LEN * 2))(cnn_output)
        return cnn_output

    # Model inputs
    input1 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
    input2 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
    input3 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
    input4 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))
    input5 = Input(shape=(SPECTROGRAM_FREQS, SPECTROGRAM_LEN, 2))

    # Build CNN outputs
    cnn_output1 = build_cnn(input1)
    cnn_output2 = build_cnn(input2)
    cnn_output3 = build_cnn(input3)
    cnn_output4 = build_cnn(input4)
    cnn_output5 = build_cnn(input5)

    # Combine CNN outputs
    cnn_concat = Concatenate(axis=1)([cnn_output1, cnn_output2, cnn_output3, cnn_output4, cnn_output5])

    # LSTM layers
    lstm_output = Bidirectional(LSTM(15, return_sequences=True, kernel_regularizer=l2(0.01)))(cnn_concat)
    output = Dense(NUM_OUTPUT_CLASSES, activation='softmax')(lstm_output)

    # Create and compile model
    model = Model(inputs=[input1, input2, input3, input4, input5], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'],
                  optimizer='adam')

    return model


def process_edf_files(psg_file, hypno_file):
    """Process uploaded EDF files and return the dataframe with EEG signals and labels"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_psg:
        tmp_psg.write(psg_file.getvalue())
        psg_path = tmp_psg.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_hypno:
        tmp_hypno.write(hypno_file.getvalue())
        hypno_path = tmp_hypno.name

    try:
        # Load the files
        try:
            psg_signals, psg_signal_headers, _ = highlevel.read_edf(psg_path)
            hypnogram_signals, _, hypnogram_header = highlevel.read_edf(hypno_path)
        except Exception as e:
            st.error(f"Error reading EDF files: {str(e)}")
            return None

        # Debug information
        # st.write("PSG File Information:")
        # for i, header in enumerate(psg_signal_headers):
        #     st.write(f"Channel {i+1}: {header['label']}")
        #     st.write(f"- Sample Rate: {header['sample_rate']} Hz")
        #     st.write(f"- Samples: {len(psg_signals[i])}")
        #     st.write(f"- Duration: {len(psg_signals[i])/header['sample_rate']:.2f} seconds")

        # Get EEG channels
        psg_df = pd.DataFrame()

        # Find the required channels
        fpz_cz_found = False
        pz_oz_found = False

        for i, header in enumerate(psg_signal_headers):
            channel_name = header['label'].lower()
            if ('fpz' in channel_name and 'cz' in channel_name) or 'eeg fpz-cz' in channel_name:
                psg_df['EEG Fpz-Cz'] = psg_signals[i]
                fpz_cz_found = True
            elif ('pz' in channel_name and 'oz' in channel_name) or 'eeg pz-oz' in channel_name:
                psg_df['EEG Pz-Oz'] = psg_signals[i]
                pz_oz_found = True

        if not (fpz_cz_found and pz_oz_found):
            st.error("Could not find required EEG channels (Fpz-Cz and Pz-Oz)")
            return None

        # Process hypnogram annotations
        sleep_stages = []
        st.write("\nHypnogram Annotations:")
        for annotation in hypnogram_header['annotations']:
            try:
                # Extract duration and stage
                dur = int(float(str(annotation[1])[2:-1]) if str(annotation[1]).startswith('(') else float(
                    str(annotation[1])))
                sleep_stage = annotation[2][-1]

                # st.write(f"Duration: {dur}s, Stage: {sleep_stage}")

                # Convert the label to integer
                if sleep_stage == 'W':
                    sleep_stage = 0
                elif sleep_stage == '1':
                    sleep_stage = 1
                elif sleep_stage == '2':
                    sleep_stage = 2
                elif sleep_stage == '3':
                    sleep_stage = 3
                elif sleep_stage == '4':
                    sleep_stage = 4
                elif sleep_stage == 'R':
                    sleep_stage = 5
                else:
                    sleep_stage = -1

                sleep_stages.extend([sleep_stage for _ in range(dur * FS)])

            except Exception as e:
                st.warning(f"Skipped invalid annotation: {str(e)}")
                continue

        # Align lengths
        if len(psg_df) > len(sleep_stages):
            st.warning(f"Trimming PSG data from {len(psg_df) / FS:.2f}s to {len(sleep_stages) / FS:.2f}s")
            psg_df = psg_df[:len(sleep_stages)]
        elif len(psg_df) < len(sleep_stages):
            st.warning(f"Trimming sleep stages from {len(sleep_stages) / FS:.2f}s to {len(psg_df) / FS:.2f}s")
            sleep_stages = sleep_stages[:len(psg_df)]

        psg_df['label'] = sleep_stages
        return psg_df

    finally:
        # Clean up temporary files
        os.unlink(psg_path)
        os.unlink(hypno_path)

    # def calculate_spectrograms(df):
    """Calculate spectrograms from the EEG signals"""
    if df is None:
        return [], []

    spectrogram_list = []
    labels_list = []

    st.write("\nProcessing Spectrograms:")
    st.progress(0.0)

    # Calculate total number of possible segments
    total_segments = (len(df) - FS * SPEC_LEN * 5) // (FS * SPEC_LEN) + 1
    processed_segments = 0

    ind = 0
    while ind + FS * SPEC_LEN * 5 <= len(df):
        spectrogram_list_tmp = []
        labels_list_tmp = []
        valid_segment = True

        # Process 5 consecutive blocks
        for i in range(5):
            try:
                # Get block data
                start_idx = ind + i * FS * SPEC_LEN
                end_idx = start_idx + FS * SPEC_LEN
                df_tmp = df.iloc[start_idx:end_idx]

                # Extract and check signals
                ch1_tmp = df_tmp['EEG Fpz-Cz'].values
                ch2_tmp = df_tmp['EEG Pz-Oz'].values

                if len(ch1_tmp) != FS * SPEC_LEN or len(ch2_tmp) != FS * SPEC_LEN:
                    valid_segment = False
                    break

                # Normalize
                if np.std(ch1_tmp) == 0 or np.std(ch2_tmp) == 0:
                    valid_segment = False
                    break

                ch1_tmp = (ch1_tmp - np.mean(ch1_tmp)) / np.std(ch1_tmp)
                ch2_tmp = (ch2_tmp - np.mean(ch2_tmp)) / np.std(ch2_tmp)

                # Calculate spectrograms
                Sxx1 = melspectrogram(y=ch1_tmp, sr=FS, n_fft=n_fft,
                                      hop_length=hop_length, n_mels=n_mels)
                Sxx1 = power_to_db(Sxx1, ref=np.max)

                Sxx2 = melspectrogram(y=ch2_tmp, sr=FS, n_fft=n_fft,
                                      hop_length=hop_length, n_mels=n_mels)
                Sxx2 = power_to_db(Sxx2, ref=np.max)

                # Get label (most frequent in this block)
                label = df_tmp['label'].mode().iloc[0]

                spectrogram_list_tmp.append((Sxx1, Sxx2))
                labels_list_tmp.append(label)

            except Exception as e:
                st.warning(f"Error processing block at {start_idx / FS:.1f}s: {str(e)}")
                valid_segment = False
                break

        # Only keep valid segments
        if valid_segment and len(spectrogram_list_tmp) == 5 and -1 not in labels_list_tmp:
            spectrogram_list.append(spectrogram_list_tmp)
            labels_list.append(labels_list_tmp)

        processed_segments += 1
        st.progress(min(processed_segments / total_segments, 1.0))

        ind += FS * SPEC_LEN

    st.write(f"Processed {len(spectrogram_list)} valid segments")

    if not spectrogram_list:
        return [], []

    # Convert labels to 3-class format
    three_class_labels = []
    for labels in labels_list:
        labels = np.array(labels)
        # Map NREM stages (1-4) to class 1
        labels[(labels >= 1) & (labels <= 4)] = 1
        # Map REM (5) to class 2
        labels[labels == 5] = 2
        three_class_labels.append(labels)

    return spectrogram_list, three_class_labels


def prepare_model_input(spectrograms):
    """Prepare spectrograms for model input"""
    if not spectrograms:
        raise ValueError("No valid spectrogram data available")

    # Initialize lists for each input
    X1, X2, X3, X4, X5 = [], [], [], [], []

    for spec_group in spectrograms:
        # Stack each channel's spectrogram
        spec1 = np.stack([spec_group[0][0], spec_group[0][1]], axis=-1)  # First block
        spec2 = np.stack([spec_group[1][0], spec_group[1][1]], axis=-1)  # Second block
        spec3 = np.stack([spec_group[2][0], spec_group[2][1]], axis=-1)  # Third block
        spec4 = np.stack([spec_group[3][0], spec_group[3][1]], axis=-1)  # Fourth block
        spec5 = np.stack([spec_group[4][0], spec_group[4][1]], axis=-1)  # Fifth block

        X1.append(spec1)
        X2.append(spec2)
        X3.append(spec3)
        X4.append(spec4)
        X5.append(spec5)

    # Convert to numpy arrays with correct shape
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)

    return [X1, X2, X3, X4, X5]


def create_prediction_dataset(model_inputs, batch_size=32):
    """Create a TensorFlow dataset for predictions"""
    # Convert inputs to dictionary format
    input_names = [f'input_{i + 1}' for i in range(5)]
    input_dict = {name: tensor for name, tensor in zip(input_names, model_inputs)}

    # Create dataset from the dictionary
    dataset = tf.data.Dataset.from_tensor_slices(input_dict)
    dataset = dataset.batch(batch_size)
    return dataset


# def predict_in_batches(model, model_inputs, batch_size=32):
#     """Make predictions in batches using a TensorFlow dataset"""
#     # Create prediction dataset
#     pred_dataset = create_prediction_dataset(model_inputs, batch_size)

#     # Make predictions
#     all_predictions = []
#     total_batches = (len(model_inputs[0]) + batch_size - 1) // batch_size

#     with st.progress(0) as progress_bar:
#         for i, batch in enumerate(pred_dataset):
#             # Convert batch to list in the correct order
#             batch_inputs = [batch[f'input_{j+1}'] for j in range(5)]
#             batch_pred = model.predict(batch_inputs, verbose=0)
#             all_predictions.append(batch_pred)
#             progress_bar.progress((i + 1) / total_batches)

#     # Combine all predictions
#     return np.vstack(all_predictions)

def plot_hypnogram(predictions, true_labels=None):
    """Plot the predicted hypnogram"""
    fig, ax = plt.subplots(figsize=(12, 6))

    zero_time = datetime.datetime(2021, 1, 1, 12, 59, 0)
    timestamps = np.array([zero_time + datetime.timedelta(seconds=i * 30) for i in range(len(predictions))])

    plt.plot(timestamps, predictions, 'b-', label='Predicted Label', linewidth=2)
    if true_labels is not None:
        plt.plot(timestamps, true_labels, 'r--', label='True Label', alpha=0.7)

    plt.legend(loc='best')
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Wake', 'NREM', 'REM'])
    plt.xlabel('Time (HH:MM)')
    plt.title('Sleep Stage Hypnogram')
    plt.tight_layout()

    return fig


def calculate_spectrograms(df):
    """Calculate spectrograms from the EEG signals"""
    if df is None:
        return [], []

    spectrogram_list = []
    labels_list = []

    st.write("\nProcessing Spectrograms:")
    progress_bar = st.progress(0.0)

    # Calculate total number of possible segments
    total_segments = (len(df) - FS * SPEC_LEN * 5) // (FS * SPEC_LEN) + 1
    processed_segments = 0

    ind = 0
    while ind + FS * SPEC_LEN * 5 <= len(df):
        spectrogram_list_tmp = []
        labels_list_tmp = []
        valid_segment = True

        # Process 5 consecutive blocks
        for i in range(5):
            try:
                # Get block data
                start_idx = ind + i * FS * SPEC_LEN
                end_idx = start_idx + FS * SPEC_LEN
                df_tmp = df.iloc[start_idx:end_idx]

                # Extract and check signals
                ch1_tmp = df_tmp['EEG Fpz-Cz'].values
                ch2_tmp = df_tmp['EEG Pz-Oz'].values

                if len(ch1_tmp) != FS * SPEC_LEN or len(ch2_tmp) != FS * SPEC_LEN:
                    valid_segment = False
                    break

                # Normalize
                if np.std(ch1_tmp) == 0 or np.std(ch2_tmp) == 0:
                    valid_segment = False
                    break

                ch1_tmp = (ch1_tmp - np.mean(ch1_tmp)) / np.std(ch1_tmp)
                ch2_tmp = (ch2_tmp - np.mean(ch2_tmp)) / np.std(ch2_tmp)

                # Calculate spectrograms
                Sxx1 = melspectrogram(y=ch1_tmp, sr=FS, n_fft=n_fft,
                                      hop_length=hop_length, n_mels=n_mels)
                Sxx1 = power_to_db(Sxx1, ref=np.max)

                Sxx2 = melspectrogram(y=ch2_tmp, sr=FS, n_fft=n_fft,
                                      hop_length=hop_length, n_mels=n_mels)
                Sxx2 = power_to_db(Sxx2, ref=np.max)

                # Get label (most frequent in this block)
                label = df_tmp['label'].mode().iloc[0]

                spectrogram_list_tmp.append((Sxx1, Sxx2))
                labels_list_tmp.append(label)

            except Exception as e:
                st.warning(f"Error processing block at {start_idx / FS:.1f}s: {str(e)}")
                valid_segment = False
                break

        # Only keep valid segments
        if valid_segment and len(spectrogram_list_tmp) == 5 and -1 not in labels_list_tmp:
            spectrogram_list.append(spectrogram_list_tmp)
            labels_list.append(labels_list_tmp)

        processed_segments += 1
        progress_bar.progress(min(processed_segments / total_segments, 1.0))

        ind += FS * SPEC_LEN

    st.write(f"Processed {len(spectrogram_list)} valid segments")

    if not spectrogram_list:
        return [], []

    # Convert labels to 3-class format
    three_class_labels = []
    for labels in labels_list:
        labels = np.array(labels)
        # Map NREM stages (1-4) to class 1
        labels[(labels >= 1) & (labels <= 4)] = 1
        # Map REM (5) to class 2
        labels[labels == 5] = 2
        three_class_labels.append(labels)

    return spectrogram_list, three_class_labels


def predict_in_batches(model, model_inputs, batch_size=32):
    """Make predictions in batches using TensorFlow tensors directly"""
    # Get total number of samples
    num_samples = len(model_inputs[0])
    total_batches = (num_samples + batch_size - 1) // batch_size
    all_predictions = []

    st.write("\nMaking predictions:")
    progress_bar = st.progress(0.0)

    # Process each batch
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        # Extract batch data for each input
        batch_inputs = {
            f'input_{j + 1}': model_inputs[j][start_idx:end_idx]
            for j in range(5)
        }

        # Convert inputs to TensorFlow tensors
        batch_tensors = [
            tf.convert_to_tensor(batch_inputs[f'input_{j + 1}'], dtype=tf.float32)
            for j in range(5)
        ]

        # Make prediction
        try:
            batch_pred = model(batch_tensors, training=False)
            all_predictions.append(batch_pred.numpy())
        except Exception as e:
            st.error(f"Prediction error on batch {i + 1}: {str(e)}")
            raise e

        progress_bar.progress((i + 1) / total_batches)

    # Combine all predictions
    if all_predictions:
        return np.vstack(all_predictions)
    else:
        raise ValueError("No predictions were generated")


def interpret_sleep_quality(wake_percent, nrem_percent, rem_percent, total_time):
    """Generate interpretation of sleep quality based on sleep statistics"""
    interpretation = []

    # Total sleep time interpretation
    if total_time < 6:
        interpretation.append("Total sleep duration is shorter than recommended (less than 6 hours).")
    elif 6 <= total_time <= 9:
        interpretation.append("Total sleep duration is within the healthy range (6-9 hours).")
    else:
        interpretation.append("Total sleep duration is longer than typical (more than 9 hours).")

    # Sleep efficiency interpretation (based on wake percentage)
    if wake_percent > 20:
        interpretation.append("Sleep efficiency is low with frequent wakefulness, which may indicate disrupted sleep.")
    elif 10 <= wake_percent <= 20:
        interpretation.append("Sleep efficiency is moderate with some wake periods.")
    else:
        interpretation.append("Sleep efficiency is good with minimal wake periods.")

    # REM sleep interpretation
    if rem_percent < 15:
        interpretation.append("REM sleep percentage is lower than typical (normally 20-25%).")
    elif 15 <= rem_percent <= 25:
        interpretation.append("REM sleep percentage is within the normal range.")
    else:
        interpretation.append("REM sleep percentage is higher than typical.")

    # Overall sleep quality assessment
    if wake_percent <= 15 and 15 <= rem_percent <= 25 and nrem_percent >= 60:
        interpretation.append(
            "Overall, this appears to be a healthy sleep pattern with good proportions of NREM and REM sleep.")
    elif wake_percent > 20 or rem_percent < 10 or nrem_percent < 50:
        interpretation.append("The sleep pattern shows some irregularities that might affect sleep quality.")
    else:
        interpretation.append(
            "The sleep pattern shows typical characteristics but with some variations from ideal ranges.")

    return " ".join(interpretation)


def find_sleep_periods(pred_labels, min_sleep_duration=15, max_wake_duration=5):
    """
    Find sleep periods while allowing for brief awakenings.

    Args:
        pred_labels: array of predicted sleep stages (0=Wake, 1=NREM, 2=REM)
        min_sleep_duration: minimum duration of sleep period in minutes
        max_wake_duration: maximum duration of wake period to consider as brief awakening (minutes)

    Returns:
        mask of sleep periods including brief awakenings
    """
    # Convert minutes to number of epochs (30-second epochs)
    min_epochs = min_sleep_duration * 2
    max_wake_epochs = max_wake_duration * 2

    # Initialize arrays
    sleep_mask = np.zeros_like(pred_labels, dtype=bool)

    # Find potential sleep periods (NREM or REM)
    is_sleep = (pred_labels == 1) | (pred_labels == 2)

    # Initialize variables for tracking sleep periods
    in_sleep_period = False
    sleep_start = 0
    wake_count = 0

    for i in range(len(pred_labels)):
        if is_sleep[i]:
            if not in_sleep_period:
                # Start of new sleep period
                sleep_start = i
                in_sleep_period = True
            wake_count = 0
        else:  # Wake period
            if in_sleep_period:
                wake_count += 1
                if wake_count > max_wake_epochs:
                    # End of sleep period
                    if (i - sleep_start) >= min_epochs:
                        sleep_mask[sleep_start:i - max_wake_epochs] = True
                    in_sleep_period = False

    # Handle last sleep period
    if in_sleep_period and (len(pred_labels) - sleep_start) >= min_epochs:
        sleep_mask[sleep_start:] = True

    return sleep_mask


def merge_close_sleep_periods(sleep_mask, max_gap=10):
    """
    Merge sleep periods that are separated by short gaps.

    Args:
        sleep_mask: boolean array indicating sleep periods
        max_gap: maximum gap in minutes to merge

    Returns:
        updated sleep mask with merged periods
    """
    max_gap_epochs = max_gap * 2  # convert minutes to epochs
    merged_mask = sleep_mask.copy()

    # Find gaps between sleep periods
    sleep_starts = np.where(np.diff(merged_mask.astype(int)) == 1)[0] + 1
    sleep_ends = np.where(np.diff(merged_mask.astype(int)) == -1)[0] + 1

    if len(sleep_starts) > 1:
        for i in range(len(sleep_starts) - 1):
            gap = sleep_starts[i + 1] - sleep_ends[i]
            if gap <= max_gap_epochs:
                merged_mask[sleep_ends[i]:sleep_starts[i + 1]] = True

    return merged_mask


def analyze_sleep_statistics(pred_labels, true_labels=None, sleep_mask=None):
    """Calculate sleep statistics, optionally using only sleep periods"""
    if sleep_mask is not None:
        # Include the mask in the analysis
        pred_labels = pred_labels[sleep_mask]
        if true_labels is not None:
            true_labels = true_labels[sleep_mask]

    total_epochs = len(pred_labels)

    # Calculate percentages
    wake_periods = pred_labels == 0
    nrem_periods = pred_labels == 1
    rem_periods = pred_labels == 2

    wake_percent = np.mean(wake_periods) * 100
    nrem_percent = np.mean(nrem_periods) * 100
    rem_percent = np.mean(rem_periods) * 100

    # Calculate time in hours
    total_time = total_epochs * 30 / 3600

    # Calculate accuracy if true labels are provided
    if true_labels is not None:
        accuracy = np.mean(pred_labels == true_labels) * 100
    else:
        accuracy = None

    return {
        'total_epochs': total_epochs,
        'accuracy': accuracy,
        'wake_percent': wake_percent,
        'nrem_percent': nrem_percent,
        'rem_percent': rem_percent,
        'total_time': total_time,
        'wake_time': (total_epochs * wake_percent / 100 * 30) / 3600,
        'nrem_time': (total_epochs * nrem_percent / 100 * 30) / 3600,
        'rem_time': (total_epochs * rem_percent / 100 * 30) / 3600
    }


def main_app():
    """Main application after login"""
    # Add logout button in the sidebar
    with st.sidebar:
        st.write(f"👋 Hello, {USERS[st.session_state['username']]['full_name']}!")
        st.button("Logout", on_click=logout)

    st.title("Sleep Stage Interpretation using XAI")
    st.write("""
    Upload your PSG and Hypnogram EDF files to analyze sleep stages.

    Requirements:
    - PSG file must contain EEG channels: EEG Fpz-Cz and EEG Pz-Oz
    - Sampling rate: 100 Hz
    - Hypnogram must contain sleep stage annotations (W, 1, 2, 3, 4, R)
    - Recording must be long enough for at least 5 consecutive 30-second blocks
    """)

    # File uploaders
    psg_file = st.file_uploader("Upload PSG EDF file", type=['edf'])
    hypno_file = st.file_uploader("Upload Hypnogram EDF file", type=['edf'])

    if psg_file and hypno_file:
        try:
            # Create a placeholder for status messages
            status_placeholder = st.empty()

            status_placeholder.text('Processing EDF files...')
            # Process files
            df = process_edf_files(psg_file, hypno_file)

            if df is None or len(df) < FS * SPEC_LEN * 5:
                st.error(f"Recording too short. Need at least {SPEC_LEN * 5} seconds of data.")
                return

            status_placeholder.text(f"Successfully loaded recording: {len(df) / FS:.1f} seconds")

            # Calculate spectrograms
            spectrograms, labels = calculate_spectrograms(df)

            if not spectrograms:
                st.error("No valid data segments found for analysis")
                return

            status_placeholder.text(f"Processed {len(spectrograms)} segments of {SPEC_LEN * 5} seconds each")

            # Prepare model input
            try:
                status_placeholder.text("Preparing model inputs...")
                model_input = prepare_model_input(spectrograms)

                # Debug input shapes
                # st.write("\nInput shapes:")
                # for i, inp in enumerate(model_input):
                #     st.write(f"Input {i+1}: {inp.shape}")

            except Exception as e:
                st.error(f"Error preparing model input: {str(e)}")
                return

            # Create and load model
            try:
                status_placeholder.text("Loading model...")
                model = create_model()

                # Name the input layers explicitly
                for i, input_layer in enumerate(model.inputs):
                    input_layer._name = f'input_{i + 1}'

                model.load_weights('model.h5')

                # # Debug model architecture
                # st.write("\nModel input layers:")
                # for i, input_layer in enumerate(model.inputs):
                #     st.write(f"Layer {i+1}: {input_layer.name} - Shape: {input_layer.shape}")

            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return

            # Make predictions
            try:
                status_placeholder.text("Making predictions...")

                # Verify input shapes match model expectations
                for i, (model_shape, input_shape) in enumerate(zip(
                        [layer.shape[1:] for layer in model.inputs],
                        [inp.shape[1:] for inp in model_input]
                )):
                    if model_shape != input_shape:
                        raise ValueError(
                            f"Shape mismatch for input {i + 1}: "
                            f"Model expects {model_shape}, got {input_shape}"
                        )

                predictions = predict_in_batches(model, model_input)
                status_placeholder.text("Analysis complete!")
                st.write(f"Predictions shape: {predictions.shape}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

            # Process results
            pred_labels = np.argmax(predictions, axis=-1).flatten()
            true_labels = np.array(labels).flatten()

            # First, show overall statistics
            st.write("### Overall Recording Statistics")
            overall_stats = analyze_sleep_statistics(pred_labels, true_labels)

            col1, col2, col3 = st.columns(3)
            col1.metric("Wake", f"{overall_stats['wake_percent']:.1f}%")
            col2.metric("NREM", f"{overall_stats['nrem_percent']:.1f}%")
            col3.metric("REM", f"{overall_stats['rem_percent']:.1f}%")

            st.write(f"Total recording time: {overall_stats['total_time']:.1f} hours")
            if overall_stats['accuracy'] is not None:
                st.write(f"Overall accuracy: {overall_stats['accuracy']:.1f}%")

            # st.write("\n### Sleep Period Analysis Settings")
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     min_sleep_duration = st.slider(
            #         "Minimum sleep period (minutes)",
            #         min_value=5,
            #         max_value=30,
            #         value=15,
            #         step=5
            #     )
            # with col2:
            #     max_wake_duration = st.slider(
            #         "Max brief awakening (minutes)",
            #         min_value=1,
            #         max_value=10,
            #         value=5,
            #         step=1
            #     )
            # with col3:
            #     max_gap = st.slider(
            #         "Max gap between periods (minutes)",
            #         min_value=5,
            #         max_value=30,
            #         value=10,
            #         step=5
            #     )
            min_sleep_duration = 15
            max_wake_duration = 10
            max_gap = 10

            # Find and analyze sleep periods
            sleep_mask = find_sleep_periods(pred_labels, min_sleep_duration, max_wake_duration)
            sleep_mask = merge_close_sleep_periods(sleep_mask, max_gap)
            sleep_stats = analyze_sleep_statistics(pred_labels, true_labels, sleep_mask)

            # Display sleep period statistics
            st.write("\n### Sleep Period Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Wake", f"{sleep_stats['wake_percent']:.1f}%",
                        f"{sleep_stats['wake_time']:.1f}h")
            col2.metric("NREM", f"{sleep_stats['nrem_percent']:.1f}%",
                        f"{sleep_stats['nrem_time']:.1f}h")
            col3.metric("REM", f"{sleep_stats['rem_percent']:.1f}%",
                        f"{sleep_stats['rem_time']:.1f}h")

            st.write(f"Total sleep time: {sleep_stats['total_time']:.1f} hours")
            # if sleep_stats['accuracy'] is not None:
            #     st.write(f"Accuracy during sleep periods: {sleep_stats['accuracy']:.1f}%")

            # Add sleep quality interpretation
            if sleep_stats['total_time'] > 2:  # Only interpret if we have enough sleep data
                st.write("\n### Sleep Quality Assessment")
                interpretation = interpret_sleep_quality(
                    sleep_stats['wake_percent'],
                    sleep_stats['nrem_percent'],
                    sleep_stats['rem_percent'],
                    sleep_stats['total_time']
                )
                st.info(interpretation)

                # Show time in each stage
                st.write("\n### Time in Each Stage")
                st.write(f"""
                - Wake: {sleep_stats['wake_time']:.1f} hours
                - NREM: {sleep_stats['nrem_time']:.1f} hours
                - REM: {sleep_stats['rem_time']:.1f} hours
                """)

                st.write("\n### Reference Ranges")
                st.write("""
                Typical ranges for healthy sleep:
                - Total sleep time: 6-9 hours
                - Wake: 5-15% of sleep time
                - NREM: 65-75% of sleep time
                - REM: 20-25% of sleep time

                Note: Brief awakenings are normal during sleep and typically make up 5-10% of total sleep time.
                """)


        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input files and try again.")


def main():
    """Main function to handle routing based on login state"""
    # Set page config
    st.set_page_config(
        page_title="Sleep Stage Classification",
        page_icon="🌙",
        layout="wide"
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Route to appropriate page based on login state
    if not st.session_state['logged_in']:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()