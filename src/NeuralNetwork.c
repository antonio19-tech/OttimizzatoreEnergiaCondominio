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
 * Funzione di attivazione utilizzata nello strato nascosto.
 * Introduce non linearità nel modello e migliora la capacità
 * di apprendere relazioni complesse tra le feature.
 */
double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

/*
 * Derivata della funzione ReLU.
 * Necessaria per il calcolo dei gradienti durante
 * la fase di backpropagation.
 */
double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/*
 * Softmax
 * Converte i valori di output (logits) in una distribuzione
 * di probabilità normalizzata (somma pari a 1).
 *
 * L’utilizzo della softmax consente di interpretare
 * l’output della rete come:
 *   P(stato | evidenze)
 *
 * Rendendo il modello un classificatore probabilistico.
 */
void softmax(double *x, int n) {

    /* Stabilizzazione numerica:
     * sottrazione del valore massimo per evitare overflow
     */
    double m = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > m) m = x[i];

    double s = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - m);
        s += x[i];
    }

    /* Gestione di casi patologici (somma nulla) */
    if (s <= 0.0) {
        for (int i = 0; i < n; i++)
            x[i] = 1.0 / (double)n;
        return;
    }

    /* Normalizzazione finale */
    for (int i = 0; i < n; i++)
        x[i] /= s;
}

/* ============================================================
 * INIZIALIZZAZIONE DEI PESI
 * ============================================================ */

/*
 * Genera un peso casuale con distribuzione uniforme
 * nell’intervallo [-0.5, 0.5].
 *
 * L’inizializzazione casuale rompe la simmetria iniziale
 * e consente un apprendimento efficace.
 */
double rand_weight(void) {
    return ((double)rand() / (double)RAND_MAX) - 0.5;
}

/* ============================================================
 * CREAZIONE E INIZIALIZZAZIONE DELLA RETE NEURALE
 * ============================================================ */

/*
 * Alloca e inizializza una rete neurale feed-forward
 * con:
 *  - uno strato nascosto,
 *  - funzione di attivazione ReLU,
 *  - softmax in uscita.
 *
 * La rete è progettata per operare come classificatore
 * probabilistico supervisionato.
 */
NeuralNetwork *nn_create(int inputs, int hidden, int outputs,
                         double lr, double l2) {

    NeuralNetwork *net = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    if (!net) return NULL;

    /* Parametri strutturali */
    net->num_inputs  = inputs;
    net->num_hidden  = hidden;
    net->num_outputs = outputs;

    /* Iperparametri di apprendimento */
    net->learning_rate = lr;
    net->l2 = l2;

    /* Vettori di attivazione e cache */
    net->hidden = (double*)calloc(hidden, sizeof(double));
    net->output = (double*)calloc(outputs, sizeof(double));
    net->hidden_input_cache = (double*)calloc(hidden, sizeof(double));

    /* Matrici dei pesi */
    net->weights_input_hidden =
        (double*)malloc(inputs * hidden * sizeof(double));

    net->weights_hidden_output =
        (double*)malloc(hidden * outputs * sizeof(double));

    /* Bias */
    net->bias_hidden = (double*)calloc(hidden, sizeof(double));
    net->bias_output = (double*)calloc(outputs, sizeof(double));

    /* Verifica allocazioni */
    if (!net->hidden || !net->output || !net->hidden_input_cache ||
        !net->weights_input_hidden || !net->weights_hidden_output ||
        !net->bias_hidden || !net->bias_output) {
        nn_free(net);
        return NULL;
    }

    /* Inizializzazione casuale dei pesi */
    for (int i = 0; i < inputs * hidden; i++)
        net->weights_input_hidden[i] = rand_weight();

    for (int i = 0; i < hidden * outputs; i++)
        net->weights_hidden_output[i] = rand_weight();

    /* Bias inizializzati a zero (scelta standard) */

    return net;
}

/* ============================================================
 * DEALLOCAZIONE DELLA RETE
 * ============================================================ */

/*
 * Libera tutta la memoria dinamica associata alla rete neurale.
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
 *   Input → Hidden → Output → Softmax
 *
 * Produce in output una distribuzione di probabilità
 * sugli stati di classificazione.
 */
void nn_forward(NeuralNetwork *net, const double *input) {

    /* ---------- Input → Hidden ---------- */
    for (int h = 0; h < net->num_hidden; h++) {
        double sum = net->bias_hidden[h];
        int base = h * net->num_inputs;

        for (int i = 0; i < net->num_inputs; i++)
            sum += input[i] * net->weights_input_hidden[base + i];

        /* Salvataggio per backpropagation */
        net->hidden_input_cache[h] = sum;

        /* Attivazione ReLU */
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

    /* ---------- Normalizzazione Softmax ---------- */
    softmax(net->output, net->num_outputs);
}

/* ============================================================
 * BACKPROPAGATION E AGGIORNAMENTO DEI PESI
 * ============================================================ */

/*
 * Addestramento supervisionato della rete tramite:
 *  - Softmax + Cross-Entropy Loss
 *  - Discesa del gradiente
 *  - Regolarizzazione L2
 */
void nn_train(NeuralNetwork *net,
              const double *input,
              const double *target) {

    /* Forward pass */
    nn_forward(net, input);

    /* ---------- Gradiente sull’output ---------- */
    /* dL/dz = y_pred - y_true */
    double *output_grad =
        (double*)malloc(net->num_outputs * sizeof(double));

    if (!output_grad) return;

    for (int o = 0; o < net->num_outputs; o++)
        output_grad[o] = net->output[o] - target[o];

    /* ---------- Gradiente sullo strato nascosto ---------- */
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

    /* ---------- Aggiornamento Hidden → Output ---------- */
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

    /* ---------- Aggiornamento Input → Hidden ---------- */
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
