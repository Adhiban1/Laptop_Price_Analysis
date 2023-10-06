FROM alpine:latest
WORKDIR /app
COPY dist/app .
COPY pickle_files pickle_files
COPY static static
COPY templates templates
COPY uniques.json .
CMD ["./app"]
EXPOSE 5000