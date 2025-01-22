def savelowest_loss(best_images, epoch, val=False):
    val_param = "val" if val else "train"
    norm_LRHSI = torch.tensor(denormalize(best_images["LRHSI"][0], best_images["LRHSI"][1], is_label=False, bathy=False)).permute(2, 0, 1).unsqueeze(0)
    norm_HRMSI = torch.tensor(denormalize(best_images["HRMSI"][0], best_images["HRMSI"][1], is_label=False, bathy=True)).unsqueeze(0)
    norm_GT = torch.tensor(denormalize(best_images["GT"][0], best_images["GT"][1], is_label=True, bathy=False)).permute(2, 0, 1).unsqueeze(0)
    norm_output_HRHSI = torch.tensor(denormalize(best_images["Predicted_HRHSI"][0], best_images["Predicted_HRHSI"][1], is_label=False, bathy=False, result=True)).permute(2, 0, 1).unsqueeze(0)
    norm_HRMSI = norm_HRMSI if norm_HRMSI.ndim == 4 else norm_HRMSI.unsqueeze(1)

    , img_metadata = read_geotiff3D(best_images["GT"][1], bathymetry=False)
    file_name = os.path.split(best_images["GT"][1])[-1]
    file_name_corr = file_name.split('.')[0]
    write_geotiff3D(os.path.join(os.getcwd(), 'data', 'test_results', f'{file_name_corr}recreation{valparam}{epoch}.tif'),
                    np.float32(norm_output_HRHSI.squeeze(0).permute(1, 2, 0).cpu().numpy()), img_metadata, False)
    smallest_batch_loss = min(pixelwise_loss_per_sample)
    if smallest_batch_loss < min_loss:
        smallest_batch_loss_index = torch.argmin(pixelwise_loss_per_sample)
        min_loss = smallest_batch_loss.item()
        LR_paths = batch[0]
        LR_paths = np.array(LR_paths)[valid_samples_LR.cpu().numpy()]
        GT_paths = batch[1]
        GT_paths = np.array(GT_paths)[valid_samples_GT.cpu().numpy()]
        HR_paths = batch[2]
        HR_paths = np.array(HR_paths)[valid_samples_HR.cpu().numpy()]
        best_images = {
            "LRHSI": [LRHSI[smallest_batch_loss_index.item()], LR_paths[smallest_batch_loss_index.item()]],
            "HRMSI": [HRMSI[smallest_batch_loss_index.item()], HR_paths[smallest_batch_loss_index.item()]],
            "GT": [GT[smallest_batch_loss_index.item()], GT_paths[smallest_batch_loss_index.item()]],
            "Predicted_HRHSI": [output_HRHSI[smallest_batch_loss_index.item()].detach(),
                                GT_paths[smallest_batch_loss_index.item()]]
        }
        min_loss = float('inf')  # Initialize to infinity
                    best_images = {}  # Dictionary to store images for lowest loss
    for idx, path in enumerate(batch[1]):
        if "410" in path:
            LR_paths = batch[0]
            LR_paths = np.array(LR_paths)[valid_samples_LR.cpu().numpy()]
            GT_paths = batch[1]
            GT_paths = np.array(GT_paths)[valid_samples_GT.cpu().numpy()]
            HR_paths = batch[2]
            HR_paths = np.array(HR_paths)[valid_samples_HR.cpu().numpy()]
            img_410 = {
                "LRHSI": [LRHSI[idx], LR_paths[idx]],
                "HRMSI": [HRMSI[idx], HR_paths[idx]],
                "GT": [GT[idx], GT_paths[idx]],
                "Predicted_HRHSI": [output_HRHSI[idx].detach(), GT_paths[idx]]
            }
            save_lowest_loss(img_410, epoch, val=True)