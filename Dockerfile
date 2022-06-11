FROM john/pyspark:3.2.1-hadoop2.7-py3.7

# ADD Python dependencies
COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# Deploy node
COPY start-spark.sh /
CMD ["/bin/bash", "/start-spark.sh"]