import java.io.*;
import java.util.LinkedList;

/**
 * Driver class for HW2 on N-grams
 */
public class TextCat {

    /**
     * Print help message
     */
    private static void help() {
        final String classname = TextCat.class.getName();
        System.out.format(
                "\nPrints the log-probability of each file under a smoothed n-gram model.\n\n" +
                        "Usage:   java %s smoother lexicon trainpath files...\n" +
                        "Example: java %s add0.01 %shw-lm/lexicons/words-10.txt switchboard-small %shw-lm/speech/sample*\n\n" +
                        LanguageModel.getLMDocumentation() +
                        "trainpath is the location of the training corpus\n" +
                        "  (the search path for this includes \"%s)\"\n\n",
                classname, classname, Constants.courseDir, Constants.courseDir, Constants.defaultTrainingDir);
    }

    /**
     * Calculates the probability of a file
     *
     * @param testfile the location of the
     * @param lm       a trained language model
     * @return log probability of a file (i.e. a sequence of words in a file)
     * @throws IOException on error reading file
     */
    public static double fileLogProb(String testfile, LanguageModel lm)
            throws IOException {
        BufferedReader reader =
                new BufferedReader(new FileReader(new File(testfile)));

        double logprob = 0.0;

        String x = Constants.BOS;
        String y = Constants.BOS;

        String line;
        while ((line = reader.readLine()) != null) {
            for (String z : line.trim().split("\\s+")) {
                if (!lm.vocab.contains(z)) z = Constants.OOV;
                logprob += Math.log(lm.prob(x, y, z));
                x = y;
                y = z;
            }
        }
        logprob += Math.log(lm.prob(x, y, Constants.EOS));
        reader.close();

        return logprob;
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            help();
            System.exit(1);
        }

        final String smoother = args[0];
        final String lexicon = args[1];
        final String trainFile = args[2];
        final String trainFile2 = args[3];
        
        LinkedList<Double> lm1Prob = new LinkedList<Double>();
        LinkedList<Double> lm2Prob = new LinkedList<Double>();

        if (args.length < 4) {
            System.err.println("warning: no input files specified");
        }

        LanguageModel lm = null;
        LanguageModel lm2 = null;
        try {
            lm = LanguageModel.getLM(smoother, lexicon);
            lm2 = LanguageModel.getLM(smoother, lexicon);
            lm.setVocabSize(trainFile, trainFile2);
            lm2.setVocabSize(trainFile, trainFile2);
            lm.train(trainFile);
            lm2.train(trainFile2);
        } catch (IOException e) {
            System.err.format("error: error reading %s\n", trainFile);
            e.printStackTrace(System.err);
            System.exit(1);
        }

        for (int i = 4; i < args.length; i++) {
            final String testfile = args[i];
            try {
                // We use natural log for our internal computations and that's
                // the kind of log-probability that fileLogProb returns.
                // But we'd like to print a value in bits: so we convert
                // log base e to log base 2 at print time, by dividing by log(2).
            	lm1Prob.add(fileLogProb(testfile, lm)/ Constants.LOG2);
            	lm2Prob.add(fileLogProb(testfile, lm2)/ Constants.LOG2);
            } catch (IOException e) {
                System.err.format("warning: error reading %s\n", testfile);
                e.printStackTrace(System.err);
            }
        }
        int counter1 = 0;
        int counter2 = 0;
        for (int i = 4; i < args.length; i++){
        	if (lm1Prob.get(i - 4) > lm2Prob.get(i - 4)){
        		System.out.println(args[2] + "  " + args[i]);
        		counter1++;
        	}else{
        		System.out.println(args[3] + "  " + args[i]);
        		counter2++;
        	}
        }
        System.out.println(counter1 + " looked more like " + args[2] + " (" + ((double)counter1 / (double)(counter1+counter2)) + ")");
        System.out.println(counter2 + " looked more like " + args[3] + " (" + ((double)counter2 / (double)(counter1+counter2)) + ")");
    }
}
