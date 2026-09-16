import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import re
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import kornia.augmentation as K_A
from kornia.constants import Resample
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import RACE_TO_ID


def imgunit16(img):
    mammogram_scaled = (
            (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535
    )
    return mammogram_scaled


def extract_date_from_filename(filename):
    """Extracts a datetime object from a standard filename."""
    # e.g., "12345_R_MLO_2019-07-13_....png" -> "2019-07-13"
    try:
        date_str = filename.split("_")[3]
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (IndexError, ValueError):
        return None  # Return None if filename format is unexpected


class BreastCancerRiskDataset_multiple_Prior_EMBED(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5,
                 max_priors=2, min_priors=0, n_priors=None, fixed_cohort=False,):
        """
        Args:
            csv_file (str): Path to the CSV file containing patient data.
            image_dir (str): Directory containing mammogram images.
            transform (callable, optional): Optional transform to be applied on an image.
            n_years (int): Number of years for risk prediction (default: 5).
            max_priors (int): Maximum number of prior exams.
            min_priors (int): Minimum number of prior exams for train/validation.
            n_priors (int, optional): Number of priors actually used for test evaluation.
        """

        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.max_priors = max_priors
        self.min_priors = min_priors
        self.n_priors = n_priors
        self.fixed_cohort = fixed_cohort
        # Ensure date columns in the CSV are in datetime format for efficient matching
        self.data["study_date_anon"] = pd.to_datetime(self.data["study_date_anon"], errors='coerce')

        self.image_data = self._load_image_data()
        self.longitudinal_samples = self._create_longitudinal_samples()

    def map_density(self, value):
        """Map density value to integer tensor suitable for GPU operations."""
        mapping = {1: 1, 2: 2, 3: 3, 4: 4}  # Map 1-4
        index = mapping.get(value, -1)  # Use -1 for 'NA' if
        return torch.tensor(index, dtype=torch.long)

    def map_cancer_type(self, value):
        """Map cancer type (0,1,2) to integer tensor suitable for GPU ops.
           Return -1 if value is missing or outside {0,1,2}.
        """
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        REFACTORED: Load image data into a nested dictionary:
        { patient_id -> { date -> { view -> filename } } }
        This structure makes it easy to find all views for a given patient on a specific date.
        """
        image_data = defaultdict(lambda: defaultdict(dict))

        # Updated regex to be more robust
        pattern = r"(\d+)_([RL])_([A-Z]+)_(\d{4}-\d{2}-\d{2})_.*\.png"

        for filename in os.listdir(os.path.join(self.data_dir, self.mode).replace("\\", "/")):
            match = re.match(pattern, filename)
            if match:
                patient_id = match.group(1)
                laterality = match.group(2)
                view_type = match.group(3)  # CC or MLO
                date_str = match.group(4)

                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                full_view = f"{laterality}_{view_type}"  # e.g., "L_CC", "R_MLO"

                image_data[patient_id][date_obj][full_view] = filename
        return image_data

    def _create_longitudinal_samples(self):
        """
        Pre-compute consecutive longitudinal samples per patient.

        Each sample consists of:
            n_priors consecutive prior examinations
            + 1 current examination

        Example with:
            screenings = [2010, 2012, 2013, 2014, 2015]
            n_priors = 2

        generates:

            [2010, 2012] -> 2013
            [2012, 2013] -> 2014
            [2013, 2014] -> 2015

        TRAIN / VALIDATION:
            Generate consecutive windows using n_priors.

        TEST:
            If fixed_cohort=True:
                The cohort is defined by max_priors.
                Only current exams with at least max_priors
                consecutive valid prior examinations are included.
                n_priors controls how many priors are actually used.

            If fixed_cohort=False:
                Generate consecutive windows using n_priors.
        """

        samples = []

        for patient_id, date_dict in self.image_data.items():

            sorted_dates = sorted(date_dict.keys())

            for laterality in ['L', 'R']:

                cc_view = f"{laterality}_CC"
                mlo_view = f"{laterality}_MLO"

                # ---------------------------------------------------------
                # Keep only exams where both views are available
                # ---------------------------------------------------------
                valid_dates = [
                    date for date in sorted_dates
                    if (cc_view in date_dict[date]and mlo_view in date_dict[date])
                ]

                # =========================================================
                # TEST SET
                # =========================================================
                if self.mode == "test":

                    if self.fixed_cohort:

                        # -----------------------------------------------------
                        # NO PRIOR IMAGES
                        # -----------------------------------------------------
                        if self.n_priors == 0:

                            # Every valid exam can be used as a current exam.
                            # There is no history requirement.
                            for current_date in valid_dates:

                                all_priors = []

                                samples.append((patient_id, laterality, current_date, all_priors))
                        # -----------------------------------------------------
                        # FIXED COHORT WITH PRIORS
                        # -----------------------------------------------------
                        else:

                            # The fixed cohort is defined using max_priors.
                            if len(valid_dates) < self.max_priors + 1:
                                continue

                            # Generate samples for current exams that have
                            # enough history for the maximum number of priors.
                            for current_idx in range(len(valid_dates)):

                                current_date = valid_dates[current_idx]

                                # All exams before current exam, newest -> oldest
                                valid_prior_dates = valid_dates[:current_idx][::-1]

                                # Current exam must have enough history for
                                # the maximum number of priors.
                                if len(valid_prior_dates) < self.max_priors:
                                    continue

                                # Take n_priors most recent exams
                                all_priors = valid_prior_dates[:self.n_priors]
                                samples.append((patient_id, laterality, current_date, all_priors))

                    # -----------------------------------------------------
                    # NON-FIXED COHORT
                    # -----------------------------------------------------

                    # FLEXIBLE COHORT
                    # ---------------------------------------------------------
                    else:

                        # Generate a sample for every valid current exam
                        for current_idx in range(len(valid_dates)):

                            current_date = valid_dates[current_idx]

                            # All previous valid exams, newest -> oldest
                            valid_prior_dates = valid_dates[:current_idx][::-1]

                            # Need at least min_priors
                            if len(valid_prior_dates) < self.min_priors:
                                continue

                            # Use up to max_priors most recent priors
                            all_priors = valid_prior_dates[:self.max_priors]

                            samples.append((patient_id, laterality, current_date, all_priors))

                # =========================================================
                # TRAIN / VALIDATION
                # =========================================================
                else:

                    # -----------------------------------------------------
                    # VARIABLE NUMBER OF PRIORS
                    #
                    # n_priors=None
                    #
                    # Use between min_priors and max_priors prior exams.
                    #
                    # Example:
                    # min_priors = 1
                    # max_priors = 2
                    #
                    # screenings = [2010, 2012, 2013, 2014]
                    #
                    # Generates:
                    #
                    # [2010]         -> 2012
                    # [2012, 2010]   -> 2013
                    # [2013, 2012]   -> 2014
                    # -----------------------------------------------------
                    if self.n_priors is None:

                        for current_idx in range(len(valid_dates)):

                            current_date = valid_dates[current_idx]

                            # Previous valid examinations, newest -> oldest
                            valid_prior_dates = (valid_dates[:current_idx][::-1])

                            # Need at least min_priors
                            if len(valid_prior_dates) < self.min_priors:
                                continue

                            # Use at most max_priors most recent priors
                            all_priors = valid_prior_dates[:self.max_priors]

                            samples.append((patient_id, laterality, current_date, all_priors))

                    # -----------------------------------------------------
                    # FIXED NUMBER OF PRIORS
                    #
                    # n_priors is an integer
                    #
                    # Example:
                    # n_priors = 2
                    #
                    # [2010, 2012] -> 2013
                    # [2012, 2013] -> 2014
                    # -----------------------------------------------------
                    else:

                        window_size = self.n_priors + 1

                        if len(valid_dates) < window_size:
                            continue

                        for start_idx in range(len(valid_dates) - window_size + 1):

                            window = valid_dates[start_idx:start_idx + window_size]

                            all_priors = window[:-1]
                            current_date = window[-1]

                            samples.append((patient_id, laterality, current_date, all_priors))

        # Shuffle samples so consecutive windows are not presented
        # consecutively during training.
        random.shuffle(samples)

        counts = [len(s[3]) for s in samples]

        for n in sorted(set(counts)):
            print(f"  {n} prior(s): {counts.count(n)} samples")

        print(f"Found {len(samples)} longitudinal samples total.")

        return samples


    def __len__(self):
        return len(self.longitudinal_samples)

    def _get_metadata_for_image(self, filename):
        """Helper function to find the corresponding CSV row for a given image."""
        parts = filename.split("_")
        patient_id = parts[0]
        laterality = parts[1]
        view = parts[2]
        study_date = extract_date_from_filename(filename)

        matching_row = self.data[
            (self.data["patient_id"].astype(str) == patient_id) &
            (self.data["ImageLateralityFinal"] == laterality) &
            (self.data["view"] == view) &
            (self.data["study_date_anon"] == study_date)
            ]

        if matching_row.empty:
            return None  # Return None if no match is found
        return matching_row.iloc[0]  # Return the first matching row as a Series

    def __getitem__(self, idx):

        patient_id, laterality, current_date, all_priors  = self.longitudinal_samples[idx]

        cc_view = f"{laterality}_CC"
        mlo_view = f"{laterality}_MLO"

        def load_and_process(filename):
            img_path = os.path.join(self.data_dir, self.mode, filename)
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)
            img_tensor = torch.from_numpy(img_np).float() / 65535.0
            if self.transform:
                img_tensor = self.transform(img_tensor).squeeze(0)
            else:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        # Current images
        current_image_cc = load_and_process(self.image_data[patient_id][current_date][cc_view])
        current_image_mlo = load_and_process(self.image_data[patient_id][current_date][mlo_view])

        prior_images_cc = []
        prior_images_mlo = []
        time_gaps = []

        for prior_date in all_priors:
            prior_images_cc.append(
                load_and_process(self.image_data[patient_id][prior_date][cc_view])
            )
            prior_images_mlo.append(
                load_and_process(self.image_data[patient_id][prior_date][mlo_view])
            )

            gap = min(current_date.year - prior_date.year, self.n_years)
            time_gaps.append(torch.tensor(gap, dtype=torch.float32))

        n_priors = len(prior_images_cc)
        # Optional: random horizontal flip
        if self.mode == "train" and torch.rand(1).item() < 0.5:
            current_image_cc = torch.flip(current_image_cc, dims=[-1])
            current_image_mlo = torch.flip(current_image_mlo, dims=[-1])

            prior_images_cc = [torch.flip(img, dims=[-1]) for img in prior_images_cc]
            prior_images_mlo = [torch.flip(img, dims=[-1]) for img in prior_images_mlo]

        # Stack time gaps
        time_gaps = torch.stack(time_gaps) if len(time_gaps) > 0 else torch.tensor([])

        current_filename = self.image_data[patient_id][current_date][cc_view]
        current_metadata = self._get_metadata_for_image(current_filename)

        # Process metadata to get target, mask, etc.
        years_last_followup = current_metadata["years_last_followup"]
        time_to_cancer = current_metadata["Time_to_Cancer_Years"]
        density = self.map_density(current_metadata["density"])
        cancer_type =  self.map_cancer_type(current_metadata["path_severity"])
        race_str =  current_metadata.get("RACE_DESC", "")
        if not isinstance(race_str, str) or race_str.strip() == "":
            race_str = "Unknown"

        race_id = RACE_TO_ID.get(race_str, RACE_TO_ID["Unknown"])

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)

        time_to_cancer = time_to_cancer - 1
        any_cancer = time_to_cancer < 5

        # Initialize the sequence with zeros
        target = np.zeros(5, dtype=np.int8)

        # If the patient has cancer, mark the event year and later with 1
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            target[time_at_event:] = 1

        else:
            # If no cancer, use the last follow-up year
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5) - 1)
            event_observed = 0

        y_mask = np.array([1] * (time_at_event + 1) + [0] * (5 - (time_at_event + 1)), dtype=np.int8)
        y_mask = y_mask[:5]

        data = {
            "current_image_cc": current_image_cc,
            "current_image_mlo": current_image_mlo,
            "previous_image_cc": prior_images_cc,
            "previous_image_mlo": prior_images_mlo,
            'time_gap':time_gaps,
            'target': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
            'event_observed': torch.tensor(event_observed, dtype=torch.float32),
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density,
            'patient_id': patient_id,
            'cancer_type': cancer_type,
            'race':  torch.tensor(race_id, dtype=torch.long),
            "n_priors": n_priors,
        }

        return data
    
