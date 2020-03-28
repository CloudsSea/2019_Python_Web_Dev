FROM cloudsseas/anaconda_thesis_env:v1.0
COPY www /app
WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["/opt/conda/envs/thesis/bin/python3"]
CMD ["app.py"]

