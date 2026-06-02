import pandas as pd
import numpy as np

def generate_resilient_dataset(num_samples=3000):
    np.random.seed(42)
    n_per_class = num_samples // 3

    # Feature ranges with explicit geometric overlap bridging logic to prevent 100% boundary locks
    def create_class_data(label, ear_mu, ear_std, mouth_mu, mouth_std, tilt_mu, tilt_std, eye_time_mu, eye_time_std, move_mu, move_std):
        
        # Base distributions inherently generating native variance limits
        ear = np.random.normal(ear_mu, ear_std, n_per_class)
        mouth = np.random.normal(mouth_mu, mouth_std, n_per_class)
        tilt = np.random.normal(tilt_mu, tilt_std, n_per_class)
        eye_time = np.random.normal(eye_time_mu, eye_time_std, n_per_class)
        move_time = np.random.normal(move_mu, move_std, n_per_class)
        
        return pd.DataFrame({
            'EAR': ear,
            'mouth_ratio': mouth,
            'head_tilt': tilt,
            'eye_closure_time': eye_time,
            'no_movement_time': move_time,
            'label': label
        })

    # Alert: EAR high, mouth low-mid
    alert_df = create_class_data('alert', 0.28, 0.04, 0.3, 0.1, 5.0, 3.0, 0.2, 0.3, 0.5, 0.5)

    # Drowsy: Moderate EAR (overlapping alert), high mouth, moderate tilt
    drowsy_df = create_class_data('drowsy', 0.21, 0.05, 0.65, 0.15, 12.0, 5.0, 1.8, 1.0, 3.0, 1.5)

    # Unconscious: Very low EAR, moderate to high mouth, high tilt
    unconscious_df = create_class_data('unconscious', 0.14, 0.04, 0.4, 0.2, 25.0, 8.0, 5.0, 2.0, 8.0, 3.0)

    df = pd.concat([alert_df, drowsy_df, unconscious_df], ignore_index=True)
    
    # Generate 5% intense system Outliers tracking simulating hardware failure bounds
    outlier_count = int(num_samples * 0.05)
    outlier_indices = np.random.choice(df.index, outlier_count, replace=False)
    for idx in outlier_indices:
        df.loc[idx, 'EAR'] = np.random.uniform(0.0, 0.45)
        df.loc[idx, 'mouth_ratio'] = np.random.uniform(0.0, 1.0)
        df.loc[idx, 'head_tilt'] = np.random.uniform(0.0, 40.0)
        df.loc[idx, 'eye_closure_time'] = np.random.uniform(0.0, 8.0)
        df.loc[idx, 'no_movement_time'] = np.random.uniform(0.0, 10.0)
        
    df['EAR'] = df['EAR'].clip(0.05, 0.45).round(3)
    df['mouth_ratio'] = df['mouth_ratio'].clip(0.0, 1.0).round(3)
    df['head_tilt'] = df['head_tilt'].abs().round(1)
    df['eye_closure_time'] = df['eye_closure_time'].clip(0, 15).round(2)
    df['no_movement_time'] = df['no_movement_time'].clip(0, 25).round(2)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("driver_safety_resilient.csv", index=False)
    
    print(f"[\u2714] Dataset generated ({len(df)} bounds) targeting Real-World Noises and Edge Overlaps.")
    return df

if __name__ == "__main__":
    df = generate_resilient_dataset(3000)