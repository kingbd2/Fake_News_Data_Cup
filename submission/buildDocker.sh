#!/bin/bash
docker build -t sample_submission .
docker run -v ~/Projects/poetry/Fake_News_Data_Cup/data/:/usr/local/dataset/:ro --name sample sample_submission
docker rm sample