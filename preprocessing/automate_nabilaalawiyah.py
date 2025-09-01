import os
import pandas as pd
import pickle 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(path):
    try:
        # Load data
        df = pd.read_csv(path)

        # Hapus duplikat
        df.drop_duplicates(inplace=True)

        # Hapus kolom yang tidak relevan
        df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

        # One-hot encoding
        categorical_cols = ['Gender', 'Geography', 'Card Type']
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Scaling fitur numerik
        scaler = StandardScaler()
        scaler_col = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Point Earned']
        df[scaler_col] = scaler.fit_transform(df[scaler_col])

        # Deteksi dan penanganan outlier dengan IQR
        def remove_outliers_iqr(df, cols):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                before = df.shape[0]
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df
        
        df = remove_outliers_iqr(df, scaler_col)

        # Memisahkan fitur dan target
        X = df.drop('Exited', axis=1)
        y = df['Exited']

        # Terapkan SMOTE untuk menangani masalah ketidakseimbangan kelas target
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

        # Buat folder output
        output_dir = "dataset_preprocessing"
        os.makedirs(output_dir, exist_ok=True)
        print("✅ Folder dibuat:", os.path.abspath(output_dir))

        # Simpan scaler
        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # Simpan encoder
        with open(os.path.join(output_dir, "encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        # Simpan data hasil split
        pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        print("📁 Isi folder:")
        for file in os.listdir(output_dir):
            print(" -", file)

        # Simpan data gabungan hasil preprocessing
        combined_df = pd.concat([X_smote, y_smote], axis=1)
        combined_df.to_csv("bankdataset_preprocessed.csv", index=False)

        print("✅ Preprocessing selesai dan data disimpan di:", output_dir)

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print("❌ ERROR:", str(e))

if __name__ == "__main__":
    preprocess_data("Bank-Customer-Attrition-Insights-Data.csv")