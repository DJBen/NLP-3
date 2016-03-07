/**
 * The Loglinear language model
 * This is yours to implement!
 */

import java.io.*;
import java.util.*;

class LoglinearLanguageModel extends LanguageModel {

    /**
     * The word vector for w can be found at vectors.get(w).
     * You can check if a word is contained in the lexicon using
     * if w in vectors:
     */
    Map<String, double[]> vectors;  // loaded using readVectors()

    /**
     * The dimension of word vector
     */
    int dim;

    /**
     * The constant that determines the strength of the regularizer.
     * Should ordinarily be >= 0.
     */
    double C;

    /**
     * the two weight matrices U and V used in log linear model
     * They are initialized in train() function and represented as two
     * dimensional arrays.
     */
    double[][] U, V;

    /**
     * The theta value of the unigram
     */
    double beta = 0;

    double repeatWeight = 0;

    /**
     * Construct a log-linear model that is TRAINED on a particular corpus.
     *
     * @param C       The constant that determines the strength of the regularizer.
     *                Should ordinarily be >= 0.
     * @param lexicon The filename of the lexicon
     */
    public LoglinearLanguageModel(double C, String lexicon) throws java.io.IOException {
        if (C < 0) {
            System.err.println(
                    "You must include a non-negative lambda value in smoother name");
            System.exit(1);
        }
        this.C = C;
        readVectors(lexicon);
    }

    /**
     * Read word vectors from an external file.  The vectors are saved as
     * arrays in a dictionary self.vectors.
     *
     * @param filename The parameter vector: a map from feature names (strings)
     *                 to their weights.
     */
    private void readVectors(String filename) throws IOException {
        vectors = new HashMap<>();
        BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(filename)));
        String header = bufferedReader.readLine();
        String[] cfg = header.split("\\s+");
        dim = Integer.parseInt(cfg[cfg.length - 1]);
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            String[] arr = line.split("\\s+");
            assert arr.length == dim + 1;
            String word = arr[0];
            double[] vec = new double[dim];
            for (int i = 0; i < vec.length; ++i)
                vec[i] = Double.parseDouble(arr[i + 1]);
            vectors.put(word, vec);
        }

    }

    private void init() { // init model parameter
        this.U = new double[dim][dim];
        this.V = new double[dim][dim];
        for (int i = 0; i < dim; ++i) {
            Arrays.fill(U[i], 0.);
            Arrays.fill(V[i], 0.);
        }

    }
    /**
     * You probably want to call the parent method train(trainFile)
     * to collect n-gram counts, then optimize some objective function
     * that considers the n-gram counts, and finally call setTheta() on
     * the result of optimization.  See INSTRUCTIONS for more hints.
     */
    public void train(String trainFile) throws IOException {
        super.train(trainFile);
        if (U == null) init();
        double gamma0 = 0.02;  // initial learning rate, used to compute actual learning rate
        int epochs = 10;  // number of passes
        int N = tokenList.size() - 2;

        /**
         * Train the log-linear model using SGD.
         * ******** COMMENT *********
         * In log-linear model, you will have to do some additional computation at
         * this point.  You can enumerate over all training trigrams as following.
         * 
         * for (int i = 2; i < tokenList.size(); ++i) {
         *   String x = tokenList.get(i - 2);
         *   String y = tokenList.get(i - 1);
         *   String z = tokenList.get(i);
         * }
         *
         * Note2: You can use showProgress() to log progress.
         *
         **/

        System.err.println("Start optimizing.");
        int t = 0;
        for (int e = 0; e < epochs; e++) {
            double objective = 0;
            for (int i = 2; i < tokenList.size(); ++i) {
                String x = tokenList.get(i - 2);
                String y = tokenList.get(i - 1);
                String z = tokenList.get(i);
                double[] xVec = this.vector(x);
                double[] yVec = this.vector(y);
                double[] zVec = this.vector(z);
                double gamma = gamma0 / (1 + gamma0 * C / N * t);
                double thetaSquareSum = 0;
                for (int j = 0; j < U.length; j++) {
                    for (int m = 0; m < U[0].length; m++) {
                        double Zxy = this.Z(x, y);
                        double sumU = 0;
                        double sumV = 0;
                        for (String word : vocab) {
                            double prob = this.u(x, y, word) / Zxy;
                            double[] zQuoteVec = this.vector(word);
                            sumU += prob * xVec[j] * zQuoteVec[m];
                            sumV += prob * yVec[j] * zQuoteVec[m]; 
                        }

                        // Calculate F(\theta)
                        thetaSquareSum += Math.pow(U[j][m], 2) + Math.pow(V[j][m], 2);

                        // Update gradient and ascend

                        double gradientU = xVec[j] * zVec[m] - sumU - 2 * C / N * U[j][m];
                        double gradientV = yVec[j] * zVec[m] - sumV - 2 * C / N * V[j][m];
                        U[j][m] = U[j][m] + gamma * gradientU;
                        V[j][m] = V[j][m] + gamma * gradientV;
                        showProgress();
                    }
                }

                double sumBeta = 0;
                double sumRepeat = 0;
                double Zxy = this.Z(x, y);
                for (String word : vocab) {
                    double prob = this.u(x, y, word) / Zxy;
                    sumBeta += prob * this.unigramFeature(word);
                    sumRepeat += prob * this.repeatedWithin10(word);
                }
                double gradientBeta = this.unigramFeature(z) - sumBeta - 2 * C / N * beta;
                double gradientRepeat = this.repeatedWithin10(z) - sumRepeat - 2 * C / N * repeatWeight;
                beta += gamma * gradientBeta;
                repeatWeight += gamma * gradientRepeat;

                double f_i = Math.log(this.prob(x, y, z)) - C / N * thetaSquareSum;
                objective += f_i;
                t += 1;
            }
            System.out.println("Epoch " + (e + 1) + ": " + objective + "; repeat: " + repeatWeight);
        }

        System.err.format("Finished training on %d tokens", tokens.get(""));
    }

    /**
     * Computes the trigram probability p(z | x,y )
     */
    public double prob(String x, String y, String z) {
        double result = this.u(x, y, z) / this.Z(x, y);
        return result;
    }

    // Feel free to add other functions as you need.
 
    private Map<String, Integer> unigramCount = new HashMap<>();
    private double unigramFeature(String z) {
        if (unigramCount.get(z) != null) {
            return Math.log(unigramCount.get(z) + 1.0);
        }
        int count = 0;
        for (int i = 2; i < tokenList.size(); ++i) {
            String third = tokenList.get(i);
            if (third.equals(z)) {
                count++; 
            }
        }
        unigramCount.put(z, count);
        return Math.log(count + 1.0);
    }

    private Map<String, Boolean> repeatedBefore = new HashMap<>();
    private double repeatedWithin10(String z) {
        if (repeatedBefore.get(z) != null) {
            return repeatedBefore.get(z) ? 1 : 0;
        }
        for (int i = 2; i < tokenList.size(); i++) {
            if (z.equals(tokenList.get(i))) {
                for (int j = i - 1; j >= 0; j--) {
                    if (tokenList.get(j).equals(z)) {
                        repeatedBefore.put(z, true);
                        return 1;
                    }
                }
            }
        }
        repeatedBefore.put(z, false);
        return 0;
    }

    private double Z(String x, String y) {
        double sum = 0;
        for (String word : vocab) {
            sum += this.u(x, y, word);
        }
        return sum;
    }

    private double[] vector(String x) {
        return vectors.get(x) == null ? vectors.get("OOL") : vectors.get(x);
    }

    private double u(String x, String y, String z) {
        double[] xVec = this.vector(x);
        double[] yVec = this.vector(y);
        double[] zVec = this.vector(z);
        double sum = 0;
        for (int j = 0; j < U.length; j++) {
            for (int m = 0; m < U[0].length; m++) {
                sum += U[j][m] * xVec[j] * zVec[m] + V[j][m] * yVec[j] * zVec[m];
            }
        }
        sum += beta * this.unigramFeature(z);
        sum += repeatWeight * this.repeatedWithin10(z);
        return Math.exp(sum);
    }
}
