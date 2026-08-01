FROM docker.io/library/solr:9.1.1

USER root
RUN mkdir -p /opt/solr/dist/plugins && \
    wget -O /opt/solr/dist/plugins/NP-solr-dist-plugin.jar \
    https://github.com/nextprocurement/NP-solr-dist-plugin/raw/main/NP-solr-dist-plugin.jar
USER 8983
