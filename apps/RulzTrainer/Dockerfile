# RulzTrainer image for vast.ai (optional: the template works with the stock vastai/pytorch image plus
# scripts/vast/provision.sh; this bakes the install in for faster instance starts).
#
#   docker build -t <you>/rulztrainer:latest .
#   docker push <you>/rulztrainer:latest
#   then put <you>/rulztrainer:latest in the template's Image Path:Tag (see scripts/vast/create_template.sh)
#
# cuda-12.8.1-auto = current torch on CUDA 12.8: runs on RTX 30 / 40 / 50 series hosts with driver >= 570.
FROM vastai/pytorch:cuda-12.8.1-auto

# the code lands in /opt/workspace-internal and is copied to ${WORKSPACE} (/workspace) on first boot
COPY . /opt/workspace-internal/RulzTrainer
RUN . /venv/main/bin/activate && cd /opt/workspace-internal/RulzTrainer \
    && bash install.sh system \
    && install -m 0755 scripts/vast/rulztrainer.sh /opt/supervisor-scripts/rulztrainer.sh \
    && install -m 0644 scripts/vast/rulztrainer.conf /etc/supervisor/conf.d/rulztrainer.conf \
    && mkdir -p /opt/workspace-internal/RulzTrainer/datasets /opt/workspace-internal/RulzTrainer/models \
               /opt/workspace-internal/RulzTrainer/output

ENV PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:7870:17870:/:RulzTrainer|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal"
EXPOSE 7870 8080 1111 22
