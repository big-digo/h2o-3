FROM nimmis/java:14.04-oracle-8-jdk

ARG VERSION

RUN printenv
RUN echo "# Packages for Cloudera's Distribution for Hadoop, Version 5, on Ubuntu 14.04 amd64\n\
deb [arch=amd64] http://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh trusty-cdh${VERSION} contrib\n\
deb-src http://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh trusty-cdh${VERSION} contrib\n" > /etc/apt/sources.list.d/cloudera.list

# Set preference for cloudera packages
COPY h2o-hadoop/scripts/docker/cdh/conf/cloudera.pref /etc/apt/preferences.d/cloudera.pref

RUN wget http://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh/archive.key -O archive.key && sudo apt-key add archive.key && \
    sudo apt-get update

RUN apt-get install -y hadoop-conf-pseudo python-dev python-pip python-scipy
RUN pip install requests --upgrade
RUN pip install colorama future tabulate pandas

RUN service hadoop-hdfs-namenode init

# Add H2O user
RUN useradd -ms /bin/bash h2o

COPY "h2o-hadoop/scripts/docker/cdh/scripts/hdfs_start" "/etc/startup/00_hdfs_start"
RUN chmod 700 "/etc/startup/00_hdfs_start"

COPY "h2o-hadoop/scripts/docker/cdh/scripts/hdfs_setup" "/etc/startup/10_hdfs_setup"
RUN chmod 700 "/etc/startup/10_hdfs_setup"

COPY "h2o-hadoop/scripts/docker/cdh/scripts/hdfs_create_h2o_home" "/etc/startup/20_hdfs_create_h2o_home"
RUN chmod 700 "/etc/startup/20_hdfs_create_h2o_home"

COPY "h2o-hadoop/scripts/docker/cdh/scripts/yarn_start" "/etc/startup/30_yarn_start"
RUN chmod 700 "/etc/startup/30_yarn_start"

COPY "h2o-hadoop/scripts/docker/cdh/scripts/init_h2o" "/etc/startup/40_init_h2o"
RUN chmod 700 "/etc/startup/40_init_h2o"

# H2O
EXPOSE 54321
# HADOOP UI
EXPOSE 8088

ENV VERSION $VERSION
ENV INIT_H2O TRUE
ENV RUN_TESTS TRUE
ENV ENTER_BASH FALSE

COPY "h2o-hadoop/scripts/docker/cdh/scripts/startup.sh" "/usr/bin/startup.sh"
RUN chmod +x /usr/bin/startup.sh

CMD "/usr/bin/startup.sh"
