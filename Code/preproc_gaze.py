# =============================================================================
# Eye-Tracking / Gaze
# source: HMD & Unreal Engine (Log Writer)
# study: Virtual Visit
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
from tqdm import tqdm

from Code.toolbox import utils

from Code import preproc_scores, preproc_ratings, preproc_behavior


def drop_consecutive_duplicates(df, subset, keep="first", times="timestamp", tolerance=0.1):
    if keep == "first":
        df = df.loc[(df[subset].shift(1) != df[subset]) | ((df[times] - df[times].shift(1)).dt.total_seconds() >= tolerance)]
    elif keep == "last":
        df = df.loc[(df[subset].shift(-1) != df[subset]) | ((df[times].shift(-1) - df[times]).dt.total_seconds() >= tolerance)]
    return df


def get_gaze(vps, filepath, wave, df_scores):
    df_gazes = pd.DataFrame()
    df_pupil = pd.DataFrame()
    df_pupil_interaction = pd.DataFrame()
    for vp in tqdm(vps):
        # vp = vps[8]
        vp = f"0{vp}" if vp < 10 else f"{vp}"
        # print(f"VP: {vp}")

        df_gazes_vp = pd.DataFrame()
        df_pupil_vp = pd.DataFrame()
        df_pupil_interaction_vp = pd.DataFrame(columns=["VP", "event", "time", "pupil"])

        try:
            files = [item for item in os.listdir(os.path.join(filepath, 'VP_' + vp)) if (item.endswith(".csv"))]
            file = [file for file in files if "gaze" in file][0]
            df_gaze = pd.read_csv(os.path.join(filepath, 'VP_' + vp, file), sep=';', decimal='.')
        except:
            print(f"no gaze file for VP {vp}")
            continue

        # Adapt timestamps
        if (pd.to_datetime(df_gaze.loc[0, "timestamp"][0:10]) > pd.Timestamp("2023-03-26")) & (pd.to_datetime(df_gaze.loc[0, "timestamp"][0:10]) < pd.Timestamp("2023-10-29")):
            df_gaze["timestamp"] = pd.to_datetime(df_gaze["timestamp"]) + timedelta(hours=2)
        else:
            df_gaze["timestamp"] = pd.to_datetime(df_gaze["timestamp"]) + timedelta(hours=1)
        df_gaze["timestamp"] = df_gaze["timestamp"].apply(lambda t: t.replace(tzinfo=None))

        # Drop Duplicates (Samples with same Timestamp)
        df_gaze = drop_consecutive_duplicates(df_gaze, subset="timestamp", keep="first", times="timestamp", tolerance=0.01).reset_index(drop=True)
        df_gaze["timedelta"] = pd.to_timedelta(df_gaze["timestamp"] - df_gaze.loc[0, "timestamp"])
        df_gaze["timedelta"] = df_gaze["timedelta"].dt.total_seconds() * 1000
        sr, fs = utils.get_sampling_rate(df_gaze["timedelta"])
        df_gaze_resampled = df_gaze.resample(f"{int(1/50 * 1000)}ms", on="timestamp").mean(numeric_only=True)
        df_gaze_resampled = df_gaze_resampled.reset_index()
        df_gaze_resampled = pd.merge_asof(df_gaze_resampled, df_gaze[["timestamp", "actor"]], on="timestamp", tolerance=timedelta(milliseconds=100))
        df_gaze_resampled["actor"] = df_gaze_resampled["actor"].fillna("")

        # Get Conditions
        try:
            df_roles = preproc_behavior.get_conditions(vp, filepath)
        except:
            print(f"no conditions file for VP {vp}")
            continue

        # Get Events
        try:
            df_events_vp, events = preproc_behavior.get_events(vp, filepath, wave, df_roles)
        except:
            print(f"no events file for VP {vp}")
            continue

        df_events_vp = df_events_vp.loc[df_events_vp["duration"] > 0]

        df_gaze_test = df_gaze_resampled.loc[(df_gaze_resampled["timestamp"] >= events["start_test"]) & (df_gaze_resampled["timestamp"] < events["start_roomrating2"])]
        df_gaze_test = drop_consecutive_duplicates(df_gaze_test, subset="timestamp", keep="first")
        df_gaze_test = df_gaze_test.loc[df_gaze_test["eye_openness"] == 1]
        for character in ["Bryan", "Emanuel", "Ettore", "Oskar"]:
            # character = "Emanuel"
            for roi, searchstring in zip(["head", "body"], ["Head", "_Char"]):
                # roi = "head"
                # searchstring = "Head"
                number = len(df_gaze_test.loc[df_gaze_test['actor'].str.contains(f"{character}{searchstring}")])
                proportion = 0 if number == 0 else number / len(df_gaze_test)

                if roi == "head":
                    switches_towards_roi = (df_gaze_test["actor"].str.contains(f"{character}Head") & (~(df_gaze_test["actor"].shift(fill_value="").str.contains(f"{character}Head")))).sum(axis=0)
                elif roi == "body":
                    switches_towards_roi = ((df_gaze_test["actor"].str.contains(f"{character}Head") | df_gaze_test["actor"].str.contains(f"{character}_Char")) &
                                            ~((df_gaze_test["actor"].shift().str.contains(f"{character}Head") | df_gaze_test["actor"].shift().str.contains(f"{character}_Char")))).sum(axis=0)

                # Save as dataframe
                df_gaze_temp = pd.DataFrame({'VP': [int(vp)],
                                             'Phase': ["Test"],
                                             'Person': [character],
                                             'Condition': [""],
                                             'ROI': [roi],
                                             'Gaze Proportion': [proportion],
                                             'Number': [number],
                                             'Switches': [switches_towards_roi]})
                df_gazes_vp = pd.concat([df_gazes_vp, df_gaze_temp])

        # Iterate through interaction phases
        for idx_row, row in df_events_vp.iterrows():
            # idx_row = 7
            # row = df_events_vp.iloc[idx_row]
            phase = row['event']
            # print(f"Phase: {phase}")
            
            # Save continuous data for interactions and clicks
            if ("Interaction" in phase) or ("Click" in phase) or (("Visible" in phase) and not ("Actor" in phase)):
                # Get start and end point of phase
                start_phase = row['timestamp']
                end_phase = row['timestamp'] + pd.to_timedelta(row['duration'], unit="S")

                # Cut gaze dataset
                df_gaze_subset = df_gaze_resampled.loc[(df_gaze_resampled["timestamp"] >= start_phase) & (df_gaze_resampled["timestamp"] < end_phase + timedelta(seconds=1))]
                # df_gaze_subset = df_gaze_subset.loc[df_gaze_subset['event'] == phase]
                if len(df_gaze_subset) == 0:
                    continue

                df_gaze_subset = drop_consecutive_duplicates(df_gaze_subset, subset="timestamp", keep="first")
                df_gaze_subset = df_gaze_subset.reset_index(drop=True)
                
                if "Interaction" in phase:
                    # df_gaze_int = df_gaze_subset.loc[df_gaze_int["eye_openness"] == 1]
                    for character in ["Bryan", "Emanuel", "Ettore", "Oskar"]:
                        # character = "Emanuel"
                        for roi, searchstring in zip(["head", "body"], ["Head", "_Char"]):
                            # roi = "head"
                            # searchstring = "Head"
                            number = len(df_gaze_subset.loc[df_gaze_subset['actor'].str.contains(f"{character}{searchstring}")])
                            proportion = 0 if number == 0 else number / len(df_gaze_subset)

                            if roi == "head":
                                switches_towards_roi = (df_gaze_subset["actor"].str.contains(f"{character}Head") & (~(df_gaze_subset["actor"].shift(fill_value="").str.contains(f"{character}Head")))).sum(axis=0)
                            elif roi == "body":
                                switches_towards_roi = ((df_gaze_subset["actor"].str.contains(f"{character}Head") | df_gaze_subset["actor"].str.contains(f"{character}_Char")) &
                                                        ~((df_gaze_subset["actor"].shift().str.contains(f"{character}Head") | df_gaze_subset["actor"].shift().str.contains(f"{character}_Char")))).sum(axis=0)

                            # Save as dataframe
                            df_gaze_temp = pd.DataFrame({'VP': [int(vp)],
                                                         'Phase': [phase],
                                                         'Person': [character],
                                                         'Condition': [""],
                                                         'ROI': [roi],
                                                         'Gaze Proportion': [proportion],
                                                         'Number': [number],
                                                         'Switches': [switches_towards_roi]})
                            df_gazes_vp = pd.concat([df_gazes_vp, df_gaze_temp])
    
                df_gaze_subset.loc[(df_gaze_subset["pupil_left"] == -1), "pupil_left"] = np.nan
                df_gaze_subset.loc[(df_gaze_subset["pupil_left"] == 0), "pupil_left"] = np.nan
                df_gaze_subset.loc[(df_gaze_subset["pupil_left"] < df_gaze_subset["pupil_left"].mean() - 2 * df_gaze_subset["pupil_left"].std()), "pupil_left"] = np.nan
                percent_missing_left = df_gaze_subset["pupil_left"].isnull().sum() / len(df_gaze_subset)
                df_gaze_subset["pupil_left"] = df_gaze_subset["pupil_left"].interpolate(method="linear", limit_direction="both")
    
                df_gaze_subset.loc[(df_gaze_subset["pupil_right"] == -1), "pupil_right"] = np.nan
                df_gaze_subset.loc[(df_gaze_subset["pupil_right"] == 0), "pupil_right"] = np.nan
                df_gaze_subset.loc[(df_gaze_subset["pupil_right"] < df_gaze_subset["pupil_right"].mean() - 2 * df_gaze_subset["pupil_right"].std()), "pupil_right"] = np.nan
                percent_missing_right = df_gaze_subset["pupil_right"].isnull().sum() / len(df_gaze_subset)
                df_gaze_subset["pupil_right"] = df_gaze_subset["pupil_right"].interpolate(method="linear", limit_direction="both")
    
                if (percent_missing_left < 0.25) and (percent_missing_right < 0.25):
                    # Filter pupil
                    pupil = df_gaze_subset[["pupil_left", "pupil_right"]].mean(axis=1).to_numpy()
                    rolloff = 12
                    lpfreq = 2
                    pupil_filtered = np.concatenate((np.repeat(pupil[0], 100), pupil, np.repeat(pupil[-1], 100)))  # zero padding
                    pupil_filtered[np.isnan(pupil_filtered)] = np.nanmean(pupil_filtered)
                    b, a = signal.butter(int(rolloff / 6), lpfreq * (1 / (sr / 2)))  # low-pass filter
                    pupil_filtered = signal.filtfilt(b, a, pupil_filtered)  # apply filter
                    pupil_filtered = pupil_filtered[100:-100]
                    df_gaze_subset["pupil_filtered"] = pupil_filtered
    
                    # Save Pupil
                    df_pupil_temp = pd.DataFrame({'VP': [int(vp)],
                                                  'Phase': [row["event"]],
                                                  'Pupil Dilation (Mean)': [df_gaze_subset['pupil_filtered'].mean()]})
                    df_pupil_vp = pd.concat([df_pupil_vp, df_pupil_temp])

                if (wave == 2) & (df_pupil_interaction_vp["event"].str.contains(phase).any()):
                    continue
                if (percent_missing_left >= 0.25) or (percent_missing_right >= 0.25):
                    continue
                start = df_gaze_subset.loc[0, "timestamp"]
                df_gaze_subset["time"] = pd.to_timedelta(df_gaze_subset["timestamp"] - start)

                # 2 Hz low-pass butterworth filter
                timestamps = np.array(df_gaze_subset["time"].dt.total_seconds() * 1000)
                sr, fs = utils.get_sampling_rate(timestamps)

                start_pupil = df_gaze_subset.loc[0, "pupil_filtered"]
                df_gaze_subset["pupil"] = df_gaze_subset["pupil_filtered"] - start_pupil

                df_gaze_subset = df_gaze_subset.set_index("time")
                df_gaze_subset = df_gaze_subset.resample("0.1S").mean(numeric_only=True)
                df_gaze_subset = df_gaze_subset.reset_index()
                df_gaze_subset["time"] = df_gaze_subset["time"].dt.total_seconds()
                df_gaze_subset["VP"] = int(vp)
                df_gaze_subset["event"] = phase
                df_gaze_subset = df_gaze_subset[["VP", "event", "time", "pupil"]]
                df_pupil_interaction_vp = pd.concat([df_pupil_interaction_vp, df_gaze_subset])

        # Add Conditions
        for idx_row, row in df_roles.iterrows():
            # idx_row = 0
            # row = df_roles.iloc[idx_row, :]
            character = row["Character"]
            role = row["Role"]
            room = row["Rooms"]

            # df_gazes_vp["Phase"] = df_gazes_vp["Phase"].str.replace(character, role.capitalize())
            df_pupil_vp["Phase"] = df_pupil_vp["Phase"].str.replace(character, role.capitalize())
            if wave == 1:
                df_pupil_vp.loc[df_pupil_vp["Phase"].str.contains(room), "Condition"] = role
            df_gazes_vp.loc[df_gazes_vp["Person"].str.contains(character), "Condition"] = role
            df_pupil_vp.loc[df_pupil_vp["Phase"].str.contains(role.capitalize()), "Condition"] = role
            df_pupil_vp.loc[df_pupil_vp["Phase"].str.contains(character), "Condition"] = role
            df_pupil_interaction_vp["event"] = df_pupil_interaction_vp["event"].str.replace(character, role.capitalize())

        df_gazes = pd.concat([df_gazes, df_gazes_vp])
        df_pupil = pd.concat([df_pupil, df_pupil_vp])
        df_pupil_interaction = pd.concat([df_pupil_interaction, df_pupil_interaction_vp])

    # Add Participant Data
    df_gazes = df_gazes.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                         'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                         'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                         'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                         'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_gazes = df_gazes.drop(columns=['ID'])

    df_pupil = df_pupil.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                         'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                         'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                         'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                         'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']], left_on="VP", right_on="ID", how="left")
    df_pupil = df_pupil.drop(columns=['ID'])

    df_pupil_interaction = df_pupil_interaction.merge(df_scores[['ID', 'gender', 'age', 'motivation', 'tiredness',
                                                                 'SSQ-pre', 'SSQ-pre-N', 'SSQ-pre-O', 'SSQ-pre-D', 'SSQ-post', 'SSQ-post-N', 'SSQ-post-O', 'SSQ-post-D', 'SSQ-diff',
                                                                 'IPQ', 'IPQ-SP', 'IPQ-ER', 'IPQ-INV', 'MPS-PP', 'MPS-SocP', 'MPS-SelfP',
                                                                 'ASI3', 'ASI3-PC', 'ASI3-CC', 'ASI3-SC', 'SPAI', 'SIAS', 'AQ-K', 'AQ-K_SI', 'AQ-K_KR', 'AQ-K_FV',
                                                                 'ISK-K_SO', 'ISK-K_OF', 'ISK-K_SSt', 'ISK-K_RE']],
                                                      left_on="VP", right_on="ID", how="left")
    df_pupil_interaction = df_pupil_interaction.drop(columns=['ID'])

    return df_gazes, df_pupil, df_pupil_interaction


if __name__ == '__main__':
    wave = 1
    dir_path = os.getcwd()
    filepath = os.path.join(dir_path, f'Data-Wave{wave}')

    if wave == 1:
        problematic_subjects = [1, 3, 12, 19, 33, 45, 46]
    elif wave == 2:
        problematic_subjects = [1, 2, 3, 4, 20, 29, 64]

    file_name = [item for item in os.listdir(filepath) if (item.endswith(".xlsx") and "raw" in item)][0]
    df_scores_raw = pd.read_excel(os.path.join(filepath, file_name))
    df_scores_raw = df_scores_raw.loc[df_scores_raw["FINISHED"] == 1]
    df_scores, problematic_subjects = preproc_scores.create_scores(df_scores_raw, problematic_subjects)

    start = 1
    vp_folder = [int(item.split("_")[1]) for item in os.listdir(filepath) if ("VP" in item)]
    end = np.max(vp_folder)
    vps = np.arange(start, end + 1)
    vps = [vp for vp in vps if not vp in problematic_subjects]

    df_ratings, problematic_subjects = preproc_ratings.create_ratings(vps, filepath, problematic_subjects, df_scores)

    vps = [vp for vp in vps if not vp in problematic_subjects]

    df_gaze, df_pupil, df_pupil_interaction = get_gaze(vps, filepath, wave, df_scores)

    df_gaze.to_csv(os.path.join(filepath, 'gaze.csv'), decimal='.', sep=';', index=False)
    df_pupil.to_csv(os.path.join(filepath, 'pupil.csv'), decimal='.', sep=';', index=False)
    df_pupil_interaction.to_csv(os.path.join(filepath, 'pupil_interaction.csv'), decimal='.', sep=';', index=False)
