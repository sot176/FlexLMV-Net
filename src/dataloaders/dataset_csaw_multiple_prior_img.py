
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from collections import defaultdict
import random
import numpy as np


def imgunit16(img):

    mammogram_scaled = (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535

    return mammogram_scaled

class BreastCancerRiskDataset_multiple_Prior_CSAWCC(Dataset):
    def __init__(self, csv_file, image_dir, mode, transforms=None,
             n_years=5, max_priors=2, min_priors=0, n_priors=None, fixed_cohort=False):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index('file_name').to_dict(orient='index')

        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode

        self.n_priors = n_priors
        self.max_priors = max_priors
        self.min_priors = min_priors
        self.fixed_cohort = fixed_cohort

        self.image_data = self._load_image_data()
        self.patient_view_pairs = self._create_longitudinal_samples()
        self.num_elements = len(self.patient_view_pairs)

    def map_density(self, value):
        mapping = {'A': 1, 'B': 2, 'C': 3}
        index = mapping.get(value, -1)  # -1 for NA or unknown values
        return torch.tensor(index, dtype=torch.long)

    def map_cancer_type(self, value):
        """Map cancer type (0,1,2) to integer tensor suitable for GPU ops.
           Return -1 if value is missing or outside {0,1,2}.
        """
        mapping = { 1: 1, 2: 2, 3:3}
        index = mapping.get(value, -1)
        return torch.tensor(index, dtype=torch.long)

    def _load_image_data(self):
        """
        Load image data into a nested dictionary:
        { patient_id -> { exam_year -> { view -> metadata } } }
        """
        image_data = defaultdict(lambda: defaultdict(dict))  # Nested defaultdict
        csv_data = self.csv_data
        image_dir_path = os.path.join(self.data_dir, self.mode).replace("\\", "/")
        for filename in os.listdir(image_dir_path):
            if filename in csv_data:
                file_info = csv_data[filename]
                # Extract required fields
                patient_id = str(file_info['patient_id'])
                exam_year = file_info['exam_year']
                laterality = file_info['laterality']  # Left or Right
                view = file_info['viewposition']  # CC or MLO
                years_to_cancer = file_info['years_to_cancer']
                years_last_followup = file_info['years_to_last_followup']
                density = file_info['density_group']
                cancer_type = file_info['x_type']
                # Map laterality to L and R for consistency
                laterality = 'L' if laterality == 'Left' else 'R' if laterality == 'Right' else None
                # Skip if laterality is invalid
                if laterality is None:
                    print(f"Skipping file due to invalid laterality: {filename}")
                    continue
                # Group by patient_id, exam_year, and view
                key = f"{laterality}_{view}"
                image_data[patient_id][exam_year][key] = {
                    'filename': filename,
                    'years_to_cancer': years_to_cancer,
                    'years_last_followup': years_last_followup,
                    'density': density,
                    'cancer_type':cancer_type,
                }
        return image_data


    def _create_longitudinal_samples(self):
        """
        Pre-compute longitudinal samples per patient.

        Each sample consists of:

            n_priors prior examinations
            +
            1 current examination

        TRAIN / VALIDATION:
            Generate consecutive sliding windows using n_priors.

        TEST + fixed_cohort=True:
            The cohort is defined by max_priors.
            A current examination must have at least max_priors
            valid prior examinations.
            n_priors determines how many of those priors are used.

        TEST + fixed_cohort=False:
            A current examination needs at least min_priors.
            Up to max_priors most recent priors are used.
        """

        samples = []

        for patient_id, year_dict in self.image_data.items():

            sorted_years = sorted(year_dict.keys())

            for laterality in ['L', 'R']:

                cc_view = f"{laterality}_CC"
                mlo_view = f"{laterality}_MLO"

                # ---------------------------------------------------------
                # Only keep exams where BOTH CC and MLO are available
                # ---------------------------------------------------------
                valid_years = [
                    year for year in sorted_years
                    if ( cc_view in year_dict[year] and mlo_view in year_dict[year])
                ]

                # =========================================================
                # TEST SET
                # =========================================================
                if self.mode == "test":

                    # -----------------------------------------------------
                    # FIXED COHORT
                    # -----------------------------------------------------
                    if self.fixed_cohort:

                        # -------------------------------------------------
                        # NO PRIOR IMAGES
                        # -------------------------------------------------
                        if self.n_priors == 0:

                            # Every valid examination can be current exam
                            for current_year in valid_years:

                                all_priors = []

                                samples.append((patient_id, laterality, current_year, all_priors))

                        # -------------------------------------------------
                        # FIXED COHORT WITH PRIORS
                        # -------------------------------------------------
                        else:

                            # Patient must have enough examinations
                            # somewhere in their longitudinal history.
                            if len(valid_years) < self.max_priors + 1:
                                continue

                            # Every examination that has enough history
                            # for max_priors can become a current exam.
                            for current_idx in range(len(valid_years)):

                                current_year = valid_years[current_idx]

                                # Previous exams, newest -> oldest
                                valid_prior_years = (valid_years[:current_idx][::-1] )

                                # Current exam must have enough history
                                # for the fixed cohort.
                                if len(valid_prior_years) < self.max_priors:
                                    continue

                                # Use the n_priors most recent examinations
                                all_priors = valid_prior_years[:self.n_priors]

                                samples.append((patient_id, laterality, current_year, all_priors))

                    # -----------------------------------------------------
                    # NON-FIXED / FLEXIBLE COHORT
                    # -----------------------------------------------------
                    else:

                        for current_idx in range(len(valid_years)):

                            current_year = valid_years[current_idx]

                            # Previous exams, newest -> oldest
                            valid_prior_years = (valid_years[:current_idx][::-1])

                            # Need at least min_priors
                            if len(valid_prior_years) < self.min_priors:
                                continue

                            # Use up to max_priors most recent priors
                            all_priors = valid_prior_years[:self.max_priors]

                            samples.append((patient_id, laterality, current_year, all_priors))


                
                # =========================================================
                # TRAIN / VALIDATION
                # =========================================================
                else:

                    # -----------------------------------------------------
                    # VARIABLE NUMBER OF PRIORS
                    # n_priors=None
                    #
                    # Use between min_priors and max_priors prior exams.
                    #
                    # Example:
                    # max_priors = 2
                    # min_priors = 1
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

                        for current_idx in range(len(valid_years)):

                            current_year = valid_years[current_idx]

                            # All previous valid examinations,
                            # newest -> oldest
                            valid_prior_years = (valid_years[:current_idx][::-1])

                            # Need at least min_priors
                            if len(valid_prior_years) < self.min_priors:
                                continue

                            # Use at most max_priors
                            all_priors = valid_prior_years[:self.max_priors]

                            samples.append((patient_id, laterality, current_year, all_priors))

                    # -----------------------------------------------------
                    # FIXED NUMBER OF PRIORS
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

                        if len(valid_years) < window_size:
                            continue

                        for start_idx in range(len(valid_years) - window_size + 1):

                            window = valid_years[ start_idx:start_idx + window_size]

                            all_priors = window[:-1]
                            current_year = window[-1]

                            samples.append((patient_id, laterality, current_year, all_priors))
        # Shuffle
        random.shuffle(samples)

        # ---------------------------------------------------------
        # Print statistics
        # ---------------------------------------------------------
        counts = [len(s[3]) for s in samples]

        # Print number of samples for each number of priors
        for n in sorted(set(counts)):
            print(f"  {n} prior(s): {counts.count(n)} samples")

        print(f"Found {len(samples)} longitudinal samples " f"in '{self.mode}' dataset.")

        return samples

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):

        patient_id, laterality, current_year, prior_years = self.patient_view_pairs[idx]

        cc_view = f"{laterality}_CC"
        mlo_view = f"{laterality}_MLO"

        # ---------------------------------------------------------
        # Helper for loading images
        # ---------------------------------------------------------
        def load_and_process(filename):

            img_path = os.path.join(self.data_dir,self.mode,filename)

            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)
            img_tensor = torch.from_numpy(img_np).to(torch.float32)
            img_tensor = img_tensor / 256.0

            if self.transform:
                img_tensor = self.transform(img_tensor)
                img_tensor = img_tensor.squeeze(0)
            else:
                img_tensor = img_tensor.unsqueeze(0)

            return img_tensor

        # =========================================================
        # CURRENT EXAM
        # =========================================================

        current_cc_file = self.image_data[patient_id][current_year][cc_view]['filename']
        current_mlo_file = self.image_data[patient_id][current_year][mlo_view]['filename']

        current_image_cc = load_and_process(current_cc_file)
        current_image_mlo = load_and_process(current_mlo_file)

        # =========================================================
        # PRIOR EXAMS
        # =========================================================

        previous_images_cc = []
        previous_images_mlo = []

        previous_image_ids_cc = []
        previous_image_ids_mlo = []

        for prior_year in prior_years:

            prior_cc_file = (self.image_data[patient_id][prior_year][cc_view]['filename'] )
            prior_mlo_file = (self.image_data[patient_id][prior_year][mlo_view]['filename'])

            previous_images_cc.append(load_and_process(prior_cc_file))
            previous_images_mlo.append(load_and_process(prior_mlo_file))

            previous_image_ids_cc.append(prior_cc_file)
            previous_image_ids_mlo.append(prior_mlo_file)

        # =========================================================
        # TRAINING AUGMENTATION
        # =========================================================

        if self.mode == 'train':

            images = [current_image_cc,current_image_mlo]

            images.extend(previous_images_cc)
            images.extend(previous_images_mlo)

            images = torch.stack(images)

            if torch.rand(1).item() < 0.5:
                images = torch.flip(images, dims=[-1])

            current_image_cc = images[0]
            current_image_mlo = images[1]

            n_prior = len(previous_images_cc)

            previous_images_cc = list(images[2:2 + n_prior])

            previous_images_mlo = list(images[2 + n_prior:2 + 2 * n_prior])

        # =========================================================
        # METADATA
        # =========================================================

        current_metadata = (self.image_data[patient_id][current_year][cc_view])

        years_last_followup = current_metadata['years_last_followup']
        time_to_cancer = current_metadata['years_to_cancer']

        density = self.map_density(current_metadata['density'])
        cancer_type = self.map_cancer_type( current_metadata["cancer_type"])

        # =========================================================
        # TIME GAPS
        # =========================================================

        time_gaps = []

        for prior_year in prior_years:

            time_gap = abs(current_year - prior_year)

            if time_gap > 5:
                time_gap = 5

            time_gaps.append(time_gap)

        # =========================================================
        # TARGET
        # =========================================================

        if time_to_cancer == 0:
            time_to_cancer = 1

        if pd.isna(time_to_cancer):
            time_to_cancer = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)
        time_to_cancer = time_to_cancer - 1

        any_cancer = time_to_cancer < 5

        target = np.zeros(5, dtype=np.int8)

        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            target[time_at_event:] = 1

        else:

            if years_last_followup == 0:
                years_last_followup = 1

            time_at_event = int(min(years_last_followup, 5) - 1)
            event_observed = 0

        y_mask = np.array([1] * (time_at_event + 1)+ [0] * (5 - (time_at_event + 1)),dtype=np.int8)
        y_mask = y_mask[:5]

        return {
            'current_image_cc': current_image_cc, 'current_image_mlo': current_image_mlo,
            'previous_image_cc': previous_images_cc, 'previous_image_mlo': previous_images_mlo,
            'current_image_id_cc': current_cc_file, 'current_image_id_mlo': current_mlo_file,
            'previous_image_ids_cc': previous_image_ids_cc, 'previous_image_ids_mlo': previous_image_ids_mlo,
            'time_gap': torch.tensor(time_gaps, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'y_mask': torch.tensor(y_mask, dtype=torch.float32),
            'event_observed': torch.tensor(event_observed, dtype=torch.float32),
            'event_times': torch.tensor(time_at_event, dtype=torch.float32),
            'density': density, 'patient_id': torch.tensor(int(patient_id), dtype=torch.int64),
            'cancer_type': cancer_type,
        }
