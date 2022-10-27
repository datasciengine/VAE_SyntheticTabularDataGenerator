import pandas as pd


def get_reconstructed_df(vae, scaler, train_dataset_batch, COLS):
    recons_list = []
    for step, x_batch_train in enumerate(train_dataset_batch):
        recons_list.append(vae(x_batch_train))

    reconstructed_df = pd.concat([pd.DataFrame(i) for i in recons_list])
    reconstructed_df = scaler.inverse_transform(reconstructed_df)
    reconstructed_df = pd.DataFrame(reconstructed_df)
    reconstructed_df.columns = COLS

    return reconstructed_df
