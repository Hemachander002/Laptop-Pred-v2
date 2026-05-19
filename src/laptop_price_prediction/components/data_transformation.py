import os
import re
from src.laptop_price_prediction import logger
from src.laptop_price_prediction.entity.config_entity import (DataTransformationConfig)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
    
    def transform_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.config.data_path)
            data =data.dropna()
            data.columns = data.columns.str.strip()

            for col in data.select_dtypes(include='object'):
                data[col] = data[col].str.strip()

            def clean_weight(x):
                try:
                    return float(str(x).lower().replace('kg', '').strip())
                except:
                    return None
                
            data["Weight"] = data["Weight"].apply(clean_weight)
            data["Price"] = np.round(data["Price"],2)
            def clean_ram(x):
                try:
                    return float(str(x).lower().replace('gb', '').strip())
                except:
                    return None

            data["Ram"] = data["Ram"].apply(clean_ram)
            data = data.replace('?', np.nan).dropna()
            data["Inches"] = data["Inches"].astype(float)
            data["Ram"] = data["Ram"].astype(int)
            data["touch"] = data["ScreenResolution"].str.contains(r'touch\s*screen', case = False , na = False).astype(int)
            data = data[(data["Weight"] > 0.9) & (data["Weight"] < 3.6)]
            data['resolution'] = data['ScreenResolution'].str.extract(r'(\d+\s*x\s*\d+)')
            data.drop("ScreenResolution",axis = 1 , inplace = True)
            data.drop("Inches" , axis = 1 , inplace = True)
            data["amd"] = data["Cpu"].str.contains(r'\bamd\b' , case = False , na = False).astype(int)
            data["intel"] = data["Cpu"].str.contains(r'\bintel\b' , case = False , na = False).astype(int)
            def cpu_category(cpu):
                cpu = str(cpu).lower()
    
                # based on my research these are the keywords for the high end cpus
                if any(x in cpu for x in ['i7', 'i9', 'xeon', 'hq', 'hk', 'ryzen 7', 'ryzen 9', 'fx']):
                    return 'high'
    
                # these are the mid ones 
                elif any(x in cpu for x in ['i5', 'i3', 'ryzen 3', 'ryzen 5', 'a8', 'a10', 'a12']):
                    return 'mid'
    
                # rest of them are going to be considered as budget 
                else:
                    return 'budget'

            data['cpu_range'] = data['Cpu'].apply(cpu_category)
            data.drop("Cpu",inplace = True , axis = 1)
            data = pd.get_dummies(data, columns=["cpu_range"])
            def convert_storage(x):
                import re
                total = 0
    
                matches = re.findall(r'(\d+\.?\d*)(TB|GB)', str(x))
    
                for num, unit in matches:
                    num = float(num)
                    if unit == 'TB':
                        num *= 1024   # convert TB → GB
                        total += num
                    elif unit == 'GB':
                        total += num
        
                return total

            data['storage'] = data['Memory'].apply(convert_storage)
            data["ssd"] = data["Memory"].str.contains(r'\bSSD\b',case = False , na = False).astype(int)
            data["hdd"] = data["Memory"].str.contains(r'\bHDD\b', case=False, na=False).astype(int)
            data["hybrid"] = data["Memory"].str.contains(r'\bHybrid\b', case=False, na=False).astype(int)
            data["flash"] = data["Memory"].str.contains(r'\bFlash\b', case=False, na=False).astype(int)
            data["hdd+ssd"] = (
                data["Memory"].str.contains(r'\bSSD\b', case=False, na=False) &
                data["Memory"].str.contains(r'\bHDD\b', case=False, na=False)
            ).astype(int)

            data.drop("Memory",axis = 1 , inplace = True)
            data["storage"] = data["storage"].astype(int)
            data = pd.get_dummies(data, columns=['OpSys'])
            data[['width', 'height']] = data['resolution'].str.split('x', expand=True).astype(int)
            data["pixels"] = (data["height"] * data["width"])
            data.drop("resolution",axis = 1 , inplace = True)
            data = data.rename(columns={
            'OpSys_Android': "android",
            'OpSys_Chrome OS' : "chrome",
            'OpSys_Linux' : "linux", 
            'OpSys_Mac OS X' : "macX", 
            'OpSys_No OS' : "no_os",
            'OpSys_Windows 10' : "win10", 
            'OpSys_Windows 10 S' : "win10_s", 
            'OpSys_Windows 7' : "win7",
            'OpSys_macOS' : "mac_os"})
            data.columns = data.columns.str.lower()
            # now lets work on the gpu part
            import re

            def gpu_rank(gpu):
                gpu = str(gpu).lower()
                
                # INTEL  == always the low budget ones
                if 'intel' in gpu:
                    return 5
                
                # NVIDIA
                if 'nvidia' in gpu or 'geforce' in gpu:
                    
                    # RTX cards ... well 40 
                    #and 50 series cards are expensive and we dont have 
                    #the enough data to predict the 40 and 50 series laptop's price so im ignoring those cards
                    if 'rtx' in gpu:
                        num = re.findall(r'\d{2,3}', gpu)
                        if num:
                            last_two = int(num[0]) % 100
                            
                            if last_two >= 80: return 1
                            elif last_two >= 70: return 2
                            elif last_two >= 60: return 3
                            elif last_two >= 50: return 4
                        return 2  # default RTX
                    
                    # GTX cards
                    if 'gtx' in gpu:
                        num = re.findall(r'\d{1,9}', gpu)
                        if num:
                            last_two = int(num[0]) % 100
                            
                            if last_two >= 80: return 1
                            elif last_two >= 70: return 2
                            elif last_two >= 60: return 3
                            elif last_two >= 50: return 4
                            elif last_two >= 40: return 5
                            elif last_two >= 30: return 5
                    
                    # MX and GT are the low ones .. coz i was a gt user back then :( i know the pain
                    if 'mx' in gpu or 'gt' in gpu:
                        return 5
                    
                    if 'quadro' in gpu:
                        return 1
                    
                    return 5
                
                # AMD i am unsure of AMD cards coz i ve nver been an AMD consumer so based on our data. so i admit that my model works the best with the Nvidia cards
                if 'amd' in gpu or 'radeon' in gpu:
                    
                    if 'rx' in gpu:
                        num = re.findall(r'\d{5}', gpu)
                        if num:
                            last_two = int(num[0]) % 100
                            
                            if last_two >= 80: return 1
                            elif last_two >= 90: return 1
                            elif last_two >= 70: return 2
                            elif last_two >= 60: return 3
                            elif last_two >= 50: return 4
                    
                    # Old R series
                    if 'r7' in gpu: return 3
                    if 'r5' in gpu: return 4
                    if 'r3' in gpu: return 5
                    if 'r2' in gpu: return 5
                    if 'r4' in gpu: return 5

                    if 'firepro' in gpu: return 1
                    
                    return 4
                
                return 5
            
            data['gpu_rank'] = data['gpu'].apply(gpu_rank)
            data.drop("gpu",axis = 1 , inplace = True)
            data.rename(columns={"amd": "amd_cpu", "intel": "intel_cpu"}, inplace=True)
            data.drop(["amd_cpu","cpu_range_budget"],axis = 1,inplace= True)
            data.drop(["height","width"],axis = 1,inplace= True)
            data = pd.get_dummies(data,columns=["company","typename"])
            data.rename(columns = {'company_Acer' :'acer', 
                'company_Apple' : 'apple',
                'company_Asus' : "asus", 
                'company_Chuwi' : "chuwi",
                'company_Dell' : "dell", 
                'company_Fujitsu' : "fujitsu",
                'company_Google' : "google", 
                'company_HP' : "hp",
                'company_Huawei' : "huawei", 
                'company_LG' : "lg",
                'company_Lenovo' : "lenovo", 
                'company_MSI' : "msi", 
                'company_Mediacom' : "mediacom",
                'company_Microsoft' : "microsoft", 
                'company_Razer' : "razer",
                'company_Samsung' : "samsung",
                'company_Toshiba' : "toshiba",
                'company_Vero' : "vero",
                'company_Xiaomi' : "xiaomi",
                'typename_2 in 1 Convertible' : "2_in_1",
                'typename_Gaming' : "gaming",
                'typename_Netbook' : "netbook",
                'typename_Notebook' : "notebook",
                'typename_Ultrabook' : "ultrabook", 
                'typename_Workstation' : "workstation"},inplace = True)
            data.rename(columns = {"hdd+ssd" : "hdd_ssd"} , inplace = True)
            data.drop(['no_os', 'samsung', 'netbook'], axis=1, inplace=True)
            data  = data[data["price"] < data["price"].quantile(0.99)]
            bool_cols = data.select_dtypes(include='bool').columns
            data[bool_cols] = data[bool_cols].astype(int)
            logger.info("Data transformation successful!")
            return data
        except Exception as e:
            logger.exception(e)
            raise e
        
    def split_data(self,data:pd.DataFrame):
        train,test = train_test_split(data, test_size=0.2, random_state=42)
        train.to_csv(os.path.join(self.config.root_dir,"train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"), index = False)
        logger.info("Data split successful!")