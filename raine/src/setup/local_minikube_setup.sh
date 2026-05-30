#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to check if Minikube is running
minikube_status() {
    minikube status &> /dev/null
    return $?
}

# Function to wait for Kubeflow pods to be ready
wait_for_kubeflow_pods() {
    echo "Waiting for Kubeflow pods to be ready..."
    echo "This may take 10 minutes..."

    # Wait up to 1 minute for the namespace to appear
    for i in {1..12}; do
        if kubectl get namespace kubeflow &> /dev/null; then
            echo "Kubeflow namespace found."
            break
        fi
        if [ $i -eq 12 ]; then
            echo "Error: kubeflow namespace not found after 1 minute. Kubeflow may not be installed correctly."
            return 1
        fi
        echo "Waiting for kubeflow namespace to appear... (attempt $i/12)"
        sleep 5
    done

    # Wait up to 5 minutes for pods and deployments to appear and become ready
    for i in {1..60}; do
        echo "Checking Kubeflow resources (attempt $i/60)..."
        
        # Check for pods
        if kubectl get pods -n kubeflow 2>/dev/null | grep -q .; then
            if kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=10s &> /dev/null; then
                echo "All pods are ready."
                break
            fi
        fi

        # Check for deployments
        if kubectl get deployments -n kubeflow 2>/dev/null | grep -q .; then
            if kubectl wait --for=condition=Available deployments --all -n kubeflow --timeout=10s &> /dev/null; then
                echo "All deployments are available."
                break
            fi
        fi

        if [ $i -eq 60 ]; then
            echo "Error: Kubeflow resources not ready after 5 minutes."
            kubectl get pods -n kubeflow
            kubectl get deployments -n kubeflow
            return 1
        fi

        sleep 5
    done

    echo "Kubeflow setup complete."
    return 0
}

# Function to start Minikube and install Kubeflow
start_minikube() {
    # Start the Minikube cluster
    echo "Starting Minikube cluster..."
    minikube start --driver=docker --cpus 2 --memory 2048 --disk-size=10g

    # Enable addons
    echo "Enabling Minikube addons..."
    minikube addons enable storage-provisioner
    minikube addons enable default-storageclass
}

install_kubeflow() {
    # Install Kubeflow
    echo "Installing Kubeflow..."
    export PIPELINE_VERSION=2.2.0
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

    # Wait for Kubeflow pods to be ready
    wait_for_kubeflow_pods
}

# Function to set up port forwarding
setup_port_forwarding() {
    echo "Setting up port forwarding..."
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
    echo "Kubeflow is now running. You can access the UI at http://localhost:8080"
    echo "Press Ctrl+C to stop port forwarding and exit."
}

# Main script logic
if [ "$1" == "--reset" ]; then
    echo "Resetting Minikube and Kubeflow..."
    minikube delete
    # rm -rf ~/.minikube
    start_minikube
    install_kubeflow
    setup_port_forwarding
else
    if ! minikube_status; then
        echo "Minikube is not running. Starting Minikube and installing Kubeflow..."
        start_minikube
        install_kubeflow
    else
        echo "Minikube is already running. Checking Kubeflow status..."
        if ! kubectl get pods -n kubeflow &> /dev/null; then
            echo "Kubeflow is not installed. Installing Kubeflow..."
            install_kubeflow
        else
            echo "Kubeflow is already installed. Checking pod status..."
            if ! wait_for_kubeflow_pods; then
                echo "Some Kubeflow pods are not ready. Please check the pod status manually."
                exit 1
            fi
        fi
    fi
    setup_port_forwarding
fi

# Use lsof -i :8080 to figure out which pid to kill to stop port forwarding