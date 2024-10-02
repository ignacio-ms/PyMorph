img_gt = tiffreadVolume('crop_gt.tif');
img_pred = tiffreadVolume('crop_pred.tif');

img_gt = img_gt > 0;
img_pred = img_pred > 0;

similarity_og = bfscore(img_pred, img_gt);


% Erosion and dilatation

se = strel('disk', 3);
img_erode = imerode(img_pred, se);

se = strel('disk', 5);
img_dilate = imdilate(img_pred, se);

similarity_e = bfscore(img_erode, img_gt);
similarity_d = bfscore(img_dilate, img_gt);


figure;
subplot(1, 3, 1);
imshowpair(img_gt(:, :, 23), img_pred(:, :, 23));
title('original');

subplot(1, 3, 2);
imshowpair(img_gt(:, :, 23), img_erode(:, :, 23));
title('erode');

subplot(1, 3, 3);
imshowpair(img_gt(:, :, 23), img_dilate(:, :, 23));
title('dilate');
