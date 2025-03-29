# NYC Street View OCR

## Overview

This repository contains the code for the NYC Street View OCR project. It made for the workshop "Search Every Word on NYC Streets" at [NYC School of Data 2025](https://nycsodata25.sched.com) for the [NYC Open Data Week 2025](https://2025.open-data.nyc/).

The code is based on my previous open source work [nyc-gsv-collector](https://github.com/yz3440/nyc-gsv-collector) and [panoocr](https://github.com/yz3440/panoocr).

## Step 1: Find street view in a specific area

### Install scraping dependencies

```bash
pip install -r requirements-scrape.txt
```

### 1A: Sample coordinates in a specific area

```bash
python 1a-sample-coords.py
```

In the folder `geojson`, there are some geojson files that contain the area of interest. The `example.geojson` contains the adjacent area of the workshop venue.

You can also make your own geojson using this webtool [geojson.io](https://geojson.io/). You need to modify the code in `1a-sample-coords.py` to use your own geojson file.

This will sample 25m points in the area and save the results to the database.

### 1B: Search street view images around the sampled points

```bash
python 1b-search-panoramas.py
```

This will search the street view images around all the sampled points and save the results to the database.

### 1C: Add metadata to the street view images

```bash
python 1c-search-date-and-copyright.py
```

This will give you full metadata of the street view images, but it requires a Google Maps API key (which is free). Put your key in the `.env` file (you can reference the `.env.example` file).

## Step 2: OCR the street view images

### Install general dependencies

```bash
pip install -r requirements-ocr.txt
```

### Install OCR engine dependencies

Depending on your system, you might need to install the OCR engine manually, pick one of the following:

```bash
pip install -r requirements-ocr-macocr.txt

pip install -r requirements-ocr-huggingface.txt # for florence2

pip install -r requirements-ocr-paddleocr.txt # for paddleocr

pip install -r requirements-ocr-easyocr.txt # for easyocr
```

### OCR the street view images

To run the panoramic ocr, you can use the following command:

```bash
python b1-pano-ocr.py
```

This will OCR the street view images and save the results to the database.

By default, it will use the `macocr` engine. You can change it to other engines by changing the `--ocr-engine` argument. The Mac built-in OCR engine is a sweet spot between accuracy and speed.

There are some other arguments you can use:

```bash
python b1-pano-ocr.py --save-result # save the results to a `/temp` folder
```

## Step 3: Visualize the results

### Install server dependencies

```bash
pip install -r requirements-server.txt
```

### Run the server

```bash
python server.py
```

This will start a local server that you can view the results.
