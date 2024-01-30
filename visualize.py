if anomaly_map.shape != input_img.shape:
    anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
anomaly_map_norm = min_max_norm(anomaly_map)
anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

# anomaly map on image
heatmap = cvt2heatmap(anomaly_map_norm * 255)
hm_on_img = heatmap_on_image(heatmap, input_img)

# save images
cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)