#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "NeuralNetwork.h"

/* ============================================================
 * FUNZIONI DI ATTIVAZIONE
 * ============================================================ */

/*
 * ReLU (Rectified Linear Unit)
 * Usata nello strato nascosto per introdurre non linearità
 */
double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

/*
 * Derivata della ReLU
 * Necessaria per il backpropagation
 */
double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/*
 * Softmax
 * Trasforma i logits di output in una distribuzione di probabilità
 * (somma = 1)
 *
 * Questo rende la rete una:
 * 👉 RETE NEURALE PROBABILISTICA
 */
void softmax(double *x, int n) {

    // Stabilizzazione numerica (sottraggo il max)
    double m = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > m) m = x[i];

    double s = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - m);
        s += x[i];
    }

    // Caso patologico: evito divisione per zero
    if (s <= 0.0) {
        for (int i = 0; i < n; i++)
            x[i] = 1.0 / (double)n;
        return;
    }

    // Normalizzazione finale
    for (int i = 0; i < n; i++)
        x[i] /= s;
}

/* ============================================================
 * INIZIALIZZAZIONE PESI
 * ============================================================ */

/*
 * Genera un peso casuale uniforme in [-0.5, 0.5]
 * Usato per rompere la simmetria iniziale
 */
double rand_weight(void) {
    return ((double)rand() / (double)RAND_MAX) - 0.5;
}

/* ============================================================
 * CREAZIONE DELLA RETE NEURALE
 * ============================================================ */

/*
 * Alloca e inizializza una rete neurale feed-forward
 * con:
 *  - 1 hidden layer
 *  - ReLU + Softmax
 */
NeuralNetwork *nn_create(int inputs, int hidden, int outputs,
                         double lr, double l2) {

    NeuralNetwork *net = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    if (!net) return NULL;

    // Parametri strutturali
    net->num_inputs  = inputs;
    net->num_hidden  = hidden;
    net->num_outputs = outputs;

    // Iperparametri di training
    net->learning_rate = lr;
    net->l2 = l2;

    // Attivazioni e cache
    net->hidden = (double*)calloc(hidden, sizeof(double));
    net->output = (double*)calloc(outputs, sizeof(double));
    net->hidden_input_cache = (double*)calloc(hidden, sizeof(double));

    // Matrici dei pesi
    net->weights_input_hidden =
        (double*)malloc(inputs * hidden * sizeof(double));

    net->weights_hidden_output =
        (double*)malloc(hidden * outputs * sizeof(double));

    // Bias
    net->bias_hidden = (double*)calloc(hidden, sizeof(double));
    net->bias_output = (double*)calloc(outputs, sizeof(double));

    // Controllo allocazioni
    if (!net->hidden || !net->output || !net->hidden_input_cache ||
        !net->weights_input_hidden || !net->weights_hidden_output ||
        !net->bias_hidden || !net->bias_output) {
        nn_free(net);
        return NULL;
    }

    // Inizializzazione casuale dei pesi
    for (int i = 0; i < inputs * hidden; i++)
        net->weights_input_hidden[i] = rand_weight();

    for (int i = 0; i < hidden * outputs; i++)
        net->weights_hidden_output[i] = rand_weight();

    // Bias inizializzati a 0 (scelta standard)

    return net;
}

/* ============================================================
 * DEALLOCAZIONE
 * ============================================================ */

/*
 * Libera tutta la memoria associata alla rete
 */
void nn_free(NeuralNetwork *net) {
    if (!net) return;

    free(net->hidden);
    free(net->output);
    free(net->hidden_input_cache);
    free(net->weights_input_hidden);
    free(net->weights_hidden_output);
    free(net->bias_hidden);
    free(net->bias_output);
    free(net);
}

/* ============================================================
 * FORWARD PASS (INFERENZA)
 * ============================================================ */

/*
 * Esegue la propagazione in avanti:
 * Input → Hidden → Output → Softmax
 */
void nn_forward(NeuralNetwork *net, const double *input) {

    /* ---------- Input → Hidden ---------- */
    for (int h = 0; h < net->num_hidden; h++) {
        double sum = net->bias_hidden[h];
        int base = h * net->num_inputs;

        for (int i = 0; i < net->num_inputs; i++)
            sum += input[i] * net->weights_input_hidden[base + i];

        // Cache per backprop
        net->hidden_input_cache[h] = sum;

        // Attivazione ReLU
        net->hidden[h] = relu(sum);
    }

    /* ---------- Hidden → Output (logits) ---------- */
    for (int o = 0; o < net->num_outputs; o++) {
        double sum = net->bias_output[o];
        int base = o * net->num_hidden;

        for (int h = 0; h < net->num_hidden; h++)
            sum += net->hidden[h] * net->weights_hidden_output[base + h];

        net->output[o] = sum;
    }

    /* ---------- Softmax ---------- */
    softmax(net->output, net->num_outputs);
}

/* ============================================================
 * BACKPROPAGATION + AGGIORNAMENTO PESI
 * ============================================================ */

/*
 * Training con:
 *  - Softmax
 *  - Cross-Entropy Loss
 *  - Regolarizzazione L2
 */
void nn_train(NeuralNetwork *net,
              const double *input,
              const double *target) {

    // Forward pass
    nn_forward(net, input);

    /* ---------- Gradiente output ---------- */
    // dL/dz = y_pred - y_true
    double *output_grad =
        (double*)malloc(net->num_outputs * sizeof(double));

    if (!output_grad) return;

    for (int o = 0; o < net->num_outputs; o++)
        output_grad[o] = net->output[o] - target[o];

    /* ---------- Gradiente hidden ---------- */
    double *hidden_grad =
        (double*)calloc(net->num_hidden, sizeof(double));

    if (!hidden_grad) {
        free(output_grad);
        return;
    }

    for (int h = 0; h < net->num_hidden; h++) {
        double sum = 0.0;
        for (int o = 0; o < net->num_outputs; o++)
            sum += output_grad[o] *
                   net->weights_hidden_output[o * net->num_hidden + h];

        hidden_grad[h] =
            sum * relu_derivative(net->hidden_input_cache[h]);
    }

    const double lr = net->learning_rate;
    const double l2 = net->l2;

    /* ---------- Update Hidden → Output ---------- */
    for (int o = 0; o < net->num_outputs; o++) {
        for (int h = 0; h < net->num_hidden; h++) {
            int idx = o * net->num_hidden + h;
            double grad_w = output_grad[o] * net->hidden[h];
            if (l2 > 0.0)
                grad_w += l2 * net->weights_hidden_output[idx];
            net->weights_hidden_output[idx] -= lr * grad_w;
        }
        net->bias_output[o] -= lr * output_grad[o];
    }

    /* ---------- Update Input → Hidden ---------- */
    for (int h = 0; h < net->num_hidden; h++) {
        for (int i = 0; i < net->num_inputs; i++) {
            int idx = h * net->num_inputs + i;
            double grad_w = hidden_grad[h] * input[i];
            if (l2 > 0.0)
                grad_w += l2 * net->weights_input_hidden[idx];
            net->weights_input_hidden[idx] -= lr * grad_w;
        }
        net->bias_hidden[h] -= lr * hidden_grad[h];
    }

    free(output_grad);
    free(hidden_grad);
}
