from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import re


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            Path("artifacts/model_trainer/model.joblib")
        )

        self.training_cols = joblib.load(
            Path("artifacts/model_trainer/columns.joblib")
        )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    def preprocess(self, data):

        # -----------------------------------------
        # BASIC CLEANING
        # -----------------------------------------

        data.columns = data.columns.str.strip()

        for col in data.select_dtypes(include='object'):
            data[col] = data[col].str.strip()

        # -----------------------------------------
        # CLEAN WEIGHT
        # -----------------------------------------

        def clean_weight(x):
            try:
                return float(
                    str(x)
                    .lower()
                    .replace('kg', '')
                    .strip()
                )
            except:
                return None

        data["Weight"] = data["Weight"].apply(clean_weight)

        # -----------------------------------------
        # CLEAN RAM
        # -----------------------------------------

        def clean_ram(x):
            try:
                return float(
                    str(x)
                    .lower()
                    .replace('gb', '')
                    .strip()
                )
            except:
                return None

        data["Ram"] = data["Ram"].apply(clean_ram)

        # -----------------------------------------
        # TYPE CONVERSIONS
        # -----------------------------------------

        data["Inches"] = data["Inches"].astype(float)
        data["Ram"] = data["Ram"].astype(int)

        # -----------------------------------------
        # TOUCHSCREEN
        # -----------------------------------------

        data["touch"] = (
            data["ScreenResolution"]
            .str.contains(
                r'touch\s*screen',
                case=False,
                na=False
            )
            .astype(int)
        )

        # -----------------------------------------
        # RESOLUTION
        # -----------------------------------------

        data['resolution'] = data[
            'ScreenResolution'
        ].str.extract(r'(\d+\s*x\s*\d+)')

        data[['width', 'height']] = (
            data['resolution']
            .str.split('x', expand=True)
            .astype(int)
        )

        data["pixels"] = (
            data["height"] * data["width"]
        )

        # -----------------------------------------
        # CPU FEATURES
        # -----------------------------------------

        data["amd"] = (
            data["Cpu"]
            .str.contains(
                r'\bamd\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        data["intel"] = (
            data["Cpu"]
            .str.contains(
                r'\bintel\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        def cpu_category(cpu):

            cpu = str(cpu).lower()

            if any(x in cpu for x in [
                'i7',
                'i9',
                'xeon',
                'hq',
                'hk',
                'ryzen 7',
                'ryzen 9',
                'fx'
            ]):
                return 'high'

            elif any(x in cpu for x in [
                'i5',
                'i3',
                'ryzen 3',
                'ryzen 5',
                'a8',
                'a10',
                'a12'
            ]):
                return 'mid'

            else:
                return 'budget'

        data['cpu_range'] = (
            data['Cpu']
            .apply(cpu_category)
        )

        data.drop(
            "Cpu",
            inplace=True,
            axis=1
        )

        data = pd.get_dummies(
            data,
            columns=["cpu_range"]
        )

        # -----------------------------------------
        # STORAGE FEATURES
        # -----------------------------------------

        def convert_storage(x):

            total = 0

            matches = re.findall(
                r'(\d+\.?\d*)(TB|GB)',
                str(x)
            )

            for num, unit in matches:

                num = float(num)

                if unit == 'TB':
                    num *= 1024

                total += num

            return total

        data['storage'] = (
            data['Memory']
            .apply(convert_storage)
        )

        data["ssd"] = (
            data["Memory"]
            .str.contains(
                r'\bSSD\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        data["hdd"] = (
            data["Memory"]
            .str.contains(
                r'\bHDD\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        data["hybrid"] = (
            data["Memory"]
            .str.contains(
                r'\bHybrid\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        data["flash"] = (
            data["Memory"]
            .str.contains(
                r'\bFlash\b',
                case=False,
                na=False
            )
            .astype(int)
        )

        data["hdd+ssd"] = (
            data["Memory"]
            .str.contains(
                r'\bSSD\b',
                case=False,
                na=False
            )
            &
            data["Memory"]
            .str.contains(
                r'\bHDD\b',
                case=False,
                na=False
            )
        ).astype(int)

        data.drop(
            "Memory",
            axis=1,
            inplace=True
        )

        data["storage"] = (
            data["storage"]
            .astype(int)
        )

        # -----------------------------------------
        # OPERATING SYSTEM
        # -----------------------------------------

        data = pd.get_dummies(
            data,
            columns=['OpSys']
        )

        data = data.rename(columns={

            'OpSys_Android': "android",
            'OpSys_Chrome OS': "chrome",
            'OpSys_Linux': "linux",
            'OpSys_Mac OS X': "macX",
            'OpSys_No OS': "no_os",
            'OpSys_Windows 10': "win10",
            'OpSys_Windows 10 S': "win10_s",
            'OpSys_Windows 7': "win7",
            'OpSys_macOS': "mac_os"

        })

        # -----------------------------------------
        # LOWERCASE
        # -----------------------------------------

        data.columns = data.columns.str.lower()

        # -----------------------------------------
        # GPU FEATURES
        # -----------------------------------------

        def gpu_rank(gpu):

            gpu = str(gpu).lower()

            # Intel GPU
            if 'intel' in gpu:
                return 5

            # NVIDIA
            if 'nvidia' in gpu or 'geforce' in gpu:

                if 'rtx' in gpu:

                    num = re.findall(r'\d{2,3}', gpu)

                    if num:

                        last_two = int(num[0]) % 100

                        if last_two >= 80:
                            return 1

                        elif last_two >= 70:
                            return 2

                        elif last_two >= 60:
                            return 3

                        elif last_two >= 50:
                            return 4

                    return 2

                if 'gtx' in gpu:

                    num = re.findall(r'\d{1,9}', gpu)

                    if num:

                        last_two = int(num[0]) % 100

                        if last_two >= 80:
                            return 1

                        elif last_two >= 70:
                            return 2

                        elif last_two >= 60:
                            return 3

                        elif last_two >= 50:
                            return 4

                        elif last_two >= 40:
                            return 5

                    return 5

                if 'mx' in gpu or 'gt' in gpu:
                    return 5

                if 'quadro' in gpu:
                    return 1

                return 5

            # AMD GPU
            if 'amd' in gpu or 'radeon' in gpu:

                if 'firepro' in gpu:
                    return 1

                if 'r7' in gpu:
                    return 3

                if 'r5' in gpu:
                    return 4

                return 4

            return 5

        data['gpu_rank'] = (
            data['gpu']
            .apply(gpu_rank)
        )

        data.drop(
            "gpu",
            axis=1,
            inplace=True
        )

        # -----------------------------------------
        # RENAME CPU COLS
        # -----------------------------------------

        data.rename(
            columns={
                "amd": "amd_cpu",
                "intel": "intel_cpu"
            },
            inplace=True
        )

        # -----------------------------------------
        # DROP UNUSED
        # -----------------------------------------

        drop_cols = [
            "amd_cpu",
            "height",
            "width",
            "screenresolution",
            "inches",
            "resolution"
        ]

        existing_cols = [
            col for col in drop_cols
            if col in data.columns
        ]

        data.drop(
            existing_cols,
            axis=1,
            inplace=True
        )

        # -----------------------------------------
        # COMPANY + TYPENAME
        # -----------------------------------------
        data.columns = data.columns.str.lower()
        
        data = pd.get_dummies(
            data,
            columns=["company", "typename"]
        )

        data.rename(columns={

            'company_acer': 'acer',
            'company_apple': 'apple',
            'company_asus': "asus",
            'company_chuwi': "chuwi",
            'company_dell': "dell",
            'company_fujitsu': "fujitsu",
            'company_google': "google",
            'company_hp': "hp",
            'company_huawei': "huawei",
            'company_lg': "lg",
            'company_lenovo': "lenovo",
            'company_msi': "msi",
            'company_mediacom': "mediacom",
            'company_microsoft': "microsoft",
            'company_razer': "razer",
            'company_samsung': "samsung",
            'company_toshiba': "toshiba",
            'company_vero': "vero",
            'company_xiaomi': "xiaomi",

            'typename_2 in 1 convertible': "2_in_1",
            'typename_gaming': "gaming",
            'typename_netbook': "netbook",
            'typename_notebook': "notebook",
            'typename_ultrabook': "ultrabook",
            'typename_workstation': "workstation"

        }, inplace=True)

        # -----------------------------------------
        # HDD + SSD RENAME
        # -----------------------------------------

        data.rename(
            columns={
                "hdd+ssd": "hdd_ssd"
            },
            inplace=True
        )

        # -----------------------------------------
        # DROP UNUSED COLS
        # -----------------------------------------

        for col in ['no_os', 'samsung', 'netbook']:

            if col in data.columns:
                data.drop(
                    col,
                    axis=1,
                    inplace=True
                )

        # -----------------------------------------
        # BOOL → INT
        # -----------------------------------------

        bool_cols = data.select_dtypes(
            include='bool'
        ).columns

        data[bool_cols] = (
            data[bool_cols]
            .astype(int)
        )

        # =========================================
        # IMPORTANT
        # MATCH TRAINING COLUMNS
        # =========================================

        for col in self.training_cols:

            if col not in data.columns:
                data[col] = 0

        data = data[self.training_cols]

        return data

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self, data):

        processed_data = self.preprocess(data)

        prediction = self.model.predict(
            processed_data
        )

        return prediction
    