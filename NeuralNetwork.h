#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

/* ============================================================
 *              STRUTTURA DELLA RETE NEURALE
 * ============================================================
 *
 * Questa struttura rappresenta una rete neurale feed-forward
 * con:
 *  - uno strato di input
 *  - uno strato nascosto
 *  - uno strato di output
 *
 * L'output è una distribuzione di probabilità grazie
 * all'uso della funzione Softmax.
 */

typedef struct {

    /* -------------------------
     * Parametri strutturali
     * ------------------------- */

    int num_inputs;     // Numero di neuroni di input (feature)
    int num_hidden;     // Numero di neuroni nello strato nascosto
    int num_outputs;    // Numero di neuroni di output (stati/classi)

    /* -------------------------
     * Attivazioni dei neuroni
     * ------------------------- */

    double *hidden;     // Valori di attivazione dello strato hidden
    double *output;     // Output finale della rete:
                        // probabilità P(stato | osservazioni)

    /* -------------------------
     * Cache per il training
     * ------------------------- */

    double *hidden_input_cache;
    // Valori pre-attivazione dello strato hidden
    // Necessari per calcolare la derivata della ReLU
    // durante il backpropagation

    /* -------------------------
     * Pesi sinaptici
     * ------------------------- */

    double *weights_input_hidden;
    // Matrice dei pesi Input → Hidden
    // Dimensione: [num_hidden][num_inputs]

    double *weights_hidden_output;
    // Matrice dei pesi Hidden → Output
    // Dimensione: [num_outputs][num_hidden]

    /* -------------------------
     * Bias
     * ------------------------- */

    double *bias_hidden;   // Bias dei neuroni hidden
    double *bias_output;   // Bias dei neuroni di output

    /* -------------------------
     * Parametri di apprendimento
     * ------------------------- */

    double learning_rate;  // Tasso di apprendimento (η)
    double l2;             // Coefficiente di regolarizzazione L2
                           // (0 = disattivata)

} NeuralNetwork;

/* ============================================================
 *              INTERFACCIA PUBBLICA
 * ============================================================
 */

/*
 * Crea e inizializza una rete neurale
 *
 * inputs  : numero di feature di input
 * hidden  : numero di neuroni nello strato nascosto
 * outputs : numero di classi/stati di output
 * lr      : learning rate
 * l2      : regolarizzazione L2
 */
NeuralNetwork *nn_create(
    int inputs,
    int hidden,
    int outputs,
    double lr,
    double l2
);

/*
 * Dealloca tutta la memoria associata alla rete neurale
 */
void nn_free(NeuralNetwork *net);

/*
 * Forward propagation:
 * calcola P(stato | osservazioni)
 */
void nn_forward(
    NeuralNetwork *net,
    const double *input
);

/*
 * Training della rete neurale:
 * - forward propagation
 * - backpropagation
 * - aggiornamento dei pesi
 */
void nn_train(
    NeuralNetwork *net,
    const double *input,
    const double *target
);

#endif
