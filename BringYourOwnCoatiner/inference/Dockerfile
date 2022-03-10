FROM 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.6.0-cpu-py36-ubuntu16.04


# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install multi-model-server \
                                sagemaker-inference \
                                retrying

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY model_handler.py /home/model-server/model_handler.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]