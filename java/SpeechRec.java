import java.io.*;
import java.util.*;

/**
 * Driver class for HW2 on N-grams
 */
public class SpeechRec {

    /**
     * Print help message
     */
    private static void help() {
        final String classname = SpeechRec.class.getName();
        System.out.format(
                "\nPrints the log-probability of each file under a smoothed n-gram model.\n\n" +
                        "Usage:   java %s smoother lexicon trainpath files...\n" +
                        "Example: java %s add1 words-10.txt switchboard " + 
                        "/usr/local/data/cs465/hw-lm/speech/dev/easy/easy025 /usr/local/data/cs465/hw-lm/speech/dev/easy/easy034\n\n" +
                        LanguageModel.getLMDocumentation() +
                        "trainpath is the location of the training corpus\n" +
                        "  (the search path for this includes \"%s)\"\n\n",
                classname, classname, Constants.courseDir, Constants.courseDir, Constants.defaultTrainingDir);
    }

    public static double sentenceLogLikelihood(String sentence, LanguageModel lm) {
        double logprob = 0.0;

        String x = Constants.BOS;
        String y = Constants.BOS;

        for (String z : sentence.trim().split("\\s+")) {
            if (!lm.vocab.contains(z)) z = Constants.OOV;
            logprob += Math.log(lm.prob(x, y, z));
            x = y;
            y = z;
        }
        logprob += Math.log(lm.prob(x, y, Constants.EOS));

        return logprob;
    }

    /**
     * Calculates the probability of a file
     *
     * @param testfile the location of the
     * @param lm       a trained language model
     * @return log probability of a file (i.e. a sequence of words in a file)
     * @throws IOException on error reading file
     */
    public static void speechRec(List<String> testfiles, LanguageModel lm)
            throws IOException {


        double totalErrors = 0;
        double length = 0;

        for (String testfile : testfiles) {
            Scanner reader = new Scanner(new File(testfile));

            int actualLength = reader.nextInt();

            length += actualLength;
            // Discard line
            reader.nextLine();

            double bestLikelihood = Double.NEGATIVE_INFINITY;
            double chosenWordErrorRate = -1;

            while (reader.hasNextLine()) {
                if (reader.hasNextDouble() == false) {
                    break;
                }

                double wordErrorRate = reader.nextDouble();
                double logLikelihood = reader.nextDouble();
                double numberOfWords = reader.nextInt();

                double logprob = sentenceLogLikelihood(reader.nextLine(), lm) / Constants.LOG2;

                // System.out.print(logprob + " ");

                logprob += logLikelihood;

                // System.out.println(" " + logLikelihood);
                if (logprob > bestLikelihood) {
                    bestLikelihood = logprob;
                    chosenWordErrorRate = wordErrorRate;
                }
            }

            totalErrors += actualLength * chosenWordErrorRate;

            System.out.println("" + chosenWordErrorRate + "\t" + testfile);
            reader.close();
        }
        System.out.println("" + (totalErrors / length) + "\tOVERALL");

        // 10      well you need x number of dollars for the [recipients]
        // 0.200   -4547.2651240589        9       <s> well you need x number dollars for the [recipient] </s>
        // 0.500   -4544.29509586806       12      <s> well you need x number dollars for their so at the end </s>
        // 0.500   -4553.56970138759       12      <s> well you need x number dollars for the rest of the n </s>
        // 0.100   -4545.24438920497       9       <s> well you need x number dollars for the [recipients] </s>
        // 0.500   -4550.14474336052       12      <s> well you need x number dollars for the rest of the end </s>
        // 0.200   -4549.61094619539       9       <s> well you need tax number dollars for the [recipients] </s>
        // 0.300   -4550.57851366948       10      <s> well you need a fax number dollars for the [recipients] </s>
        // 0.200   -4546.8073088326        10      <s> well are you need x number dollars for the [recipients] </s>
        // 0.100   -4545.24438920497       9       <s> well you need x number dollars for the [recipients] </s>
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            help();
            System.exit(1);
        }

        final String smoother = args[0];
        final String lexicon = args[1];
        final String trainFile = args[2];

        if (args.length < 3) {
            System.err.println("warning: no input files specified");
        }

        LanguageModel lm = null;
        try {
            lm = LanguageModel.getLM(smoother, lexicon);
            lm.train(trainFile);
        } catch (IOException e) {
            System.err.format("error: error reading %s\n", trainFile);
            e.printStackTrace(System.err);
            System.exit(1);
        }

        List<String> files = new ArrayList<>();
        for (int i = 3; i < args.length; i++) {
            final String testfile = args[i];
            files.add(testfile);
        }
        try {
            speechRec(files, lm);
        } catch (IOException e) {
            System.err.format("warning: error reading %s\n", files);
            e.printStackTrace(System.err);
        }
    }
}
