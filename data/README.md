## Dataset Link

This dataset is available at: https://huggingface.co/datasets/Fransferdy/blender_foundation_open_movies_compression_decompression

## Introduction


This is a dataset for the task of super video decompression, a subset of the super resolution task. Usually Super Resolution models try to bump the resolution 2x or 4x starting from a fullhd picture, which is already a format rich in data and usually with little noise, in fact many people can barely see the difference in quality between a fullhd and a 4k picture. In many works synthetic datasets are created by adding gaussian noise or JPG compression, simulating the environment of bad Smartphone cameras, targeting the improvement of real life pictures. Here the proposal is a little different, the idea is to create a dataset for training models capable of decompressing a video(sequence of images with video compression) that has been compressed both in space(area) and time dimensions. For that purpose we have original HQ image sequences and LQ image sequences, created by using the popular HandBrake utility to create synthetic videos with mp4 compression with 0.25 frame size(area) at varying degrees of quality in time( the quality slider in Handbrake ). A model capable of solving this problem in real time would be a huge benefit for video platforms, such as youtube, netflix, crunchyroll, hulu, instagram, tiktok and others. Even without the real time constraint, a model capable of solving this problem would be very useful in recovery of lost media.

## Dataset 

In regards to ethics the creation of the dataset will only use public non copyrighted open animation movies, for that purpose Blender foundation movies where chosen, they are:

| Name | Duration | 
|---|---|
| _Caminandes 2_ Gran Dillama_ - Blender Animated Short   | 02:26 |
| Agent 327_ Operation Barbershop                         | 03:51 |
| Big Buck Bunny - Official Blender Foundation Short Film | 10:34 |
| Caminandes 3_ Llamigos - Funny 3D Animated Short        | 02:30 |
| Cosmos Laundromat                                       | 12:10 |
| Elephants_Dream                                         | 10:53 |
| Glass Half - Blender animated cartoon                   | 03:13 |
| Sintel                                                  | 14:48 |
| Spring - Blender Open Movie                             | 07:44 |
| Sprite_Fright - Blender_Open_Movie                      | 10:29 |
| Total Duration                                          | 78:37 |

## Dataset Generation Methodology

First all videos were gathered at the same quality level with 1920x1080 resolution.

Each video had a few seconds from the beginning and end trimmed, to remove black screens.

Initially removing all credits roll scenes was intended, but many credit roll scenes also had animation in them, so only a few of credit rolls scenes were also trimmed.

For each video, 15 new videos were created by compressing the original video into a 480p version (480x202 resolution, 4x reduction in space dimension) with varying degrees of quality(reduction in time dimension), going from 12(almost lossless) to 40(super compressed with 10% of the original size) with a quality level stride of 2(12,14...38,40 quality levels).

All videos were converted into a sequence of images, initially using the png format for lossless storage, however a single 10 minute video weighted 90 gigabytes, which proved this storage quality unfeasible, so instead, videos were converted into lossy WEBP images, with a quality factor of 90/100, this reduced the size of the video in images form from 90GB to 3GB.

### Compression Samples

https://files.realmsapp.top/Caminades_Video_Compression_Example.mp4

### Frame Deduplication
The original complete dataset has 110911 original quality images (14.1GB) and 1M 663665 video compressed images (32.9GB)

To avoid having repeated, or near similar frames on the dataset, we created a script that looks at the image sequences of the original videos, and creates a list of which frames to keep and which to discard, using average hashing to find near identical frames.

Steps of Average Hashing:
Resize the image
The image is scaled down to a very small, fixed size (in our case 16×16 pixels).
This drastically reduces details but keeps the overall structure and brightness.
The idea: two visually similar images will look almost the same at 16×16 resolution.
Convert to grayscale
Color is discarded because we only care about structure and brightness.
Now we have 256 grayscale values (0 = black, 255 = white).
Compute the average pixel value
Take the mean of all 256 grayscale values.
This average acts as a threshold.
Build a binary hash
For each pixel:
If pixel value ≥ average → 1
If pixel value < average → 0
This gives a sequence of 256 bits.
Store the hash
The 256 bits can be stored as a binary string, hexadecimal string, or integer.

The hashes are then compared using the Hamming distance (how many bits differ), images that are less than 20% different from an image that already exists is discarded.

With the list of keeps/discards created, we make a more diverse dataset by discarding all "near identical" frames for each of the compression levels of all image sequences. (if we decide to discard Caminandes_2_frame_0, we will discard it in all compression levels and vice versa for keeping frames)

This is necessary because our dataset had way too many images for us to process, after deduplication we went from 110911 HQ images to 6056 and from 1M 663665 compressed images to 90840.

## Repository Structure / How To Use

original.zip contains the original videos in 1920x1080 pixels.

cut.zip contains the original videos with beginnings and endings trimmed

processed.zip contains the videos downscaled(using HandbrakeCLI) to 480x202 pixels, with compression quality ranging from 12(least compressed, best) to 40 (most compressed, worst) with a stride of 2(12,14...38,40)

dedup.zip contains the ready to use final deduped dataset already in image sequences form, in WEBP format with quality 90/100, with deduplication removing frames that are at least 20% similar to others, it has two folders, hq_oq and lq. In hq_oq we have one folder for each video such as cosmoscut, in lq we have one folder for each video at each quality, such as cosmoscut_12, cosmoscut_14 and so on.
The images in each video folder, follows that folder name concatenated with the frame number, such as cosmoscut_frame_0001.webp, for compressed videos, the frame images also have the quality in the image name, such as cosmoscut_12_frame_0001.webp. To generate an X and Y dataset, you can split the file name by the "_" character, and use positions 0,1,3 for getting movie name, quality and frame id. The corresponding Y will be just movie name and frame id.

## Creating your own Images Dataset from these videos

If you want to generate your own image sequences dataset from these videos, you can use the python utilities included in the repository for that. The scripts require pillow and imagehash python packages, so install those (pip install pillow imagehash)

1. If you want to use the original videos, you will have to uncompress original.zip in the current folder and call python compressVideos.py to create the processed folder and the compressed videos.
Alternatively, just use the processed.zip already provided(which uses the start/end trimmed version of the videos)
2. In your processed folder(generated, or uncompressed from the zip file), run python toImageLQ.py, that will create the sequence videos for all compressed videos. (you may edit toImageLQ.py to change webp quality parameters)
3. In the main folder, run toImageHQ.py to create the image sequence videos for the original videos.(you may edit toImageHQ.py to change webp quality parameters)
4. Once 2 and 3 are complete, run python generateDuplicateMeta.py to create lists of which frames should be discarded, you can edit that file to change the parameters on what is considered a similar enough image to be discarded
5. Run python copyKeepFrames.py and python copyKeepFramesProcessed.py to generate the final deduped dataset.