/**
 * The backoff add-lambda language model
 * This is yours to implement!
 * @author hli62@jhu.edu (hansong li)
 */
class BackoffAddLambdaLanguageModel extends LanguageModel {
  final double lambda;

  /** 
   * Constructs an add-lambda language model trained on the given file.
   */
  
  public BackoffAddLambdaLanguageModel(double lambda) throws java.io.IOException {
    if (lambda < 0) {
      System.err.println(
          "You must include a non-negative lambda value in smoother name");
      System.exit(1);
    }
    this.lambda = lambda;
  }
  
  /**
   * Computes the trigram probability p(z | x,y )
   * This is yours to implement!
   */
  public double prob(final String x_, final String y_, final String z_) {
	    // Replace out-of-vocabulary words with OOV symbol.
	    String x = vocab.contains(x_) ? x_ : Constants.OOV;
	    String y = vocab.contains(y_) ? y_ : Constants.OOV;
	    String z = vocab.contains(z_) ? z_ : Constants.OOV;

	    final String xyz = x + " " + y + " " + z;
	    final String xy = x + " "  + y;
	    final String yz = y + " "  + z;
	    
	    final double xyzCount;
	    Integer xyzCountInt = tokens.get(xyz);
	    if (xyzCountInt == null) {
	      xyzCount = 0.0;
	    } else {
	      xyzCount = (double) xyzCountInt;
	    }
	    
	    final double xyCount;
	    Integer xyCountInt = tokens.get(xy);
	    if (xyCountInt == null) {
	      xyCount = 0.0;
	    } else {
	      xyCount = (double) xyCountInt;
	    }
	    
	    final double yzCount;
	    Integer yzCountInt = tokens.get(yz);
	    if (yzCountInt == null) {
	      yzCount = 0.0;
	    } else {
	      yzCount = (double) yzCountInt;
	    }
	    
	    final double zCount;
	    Integer zCountInt = tokens.get(z);
	    if (zCountInt == null) {
	      zCount = 0.0;
	    } else {
	      zCount = (double) zCountInt;
	    }
	    
	    final double yCount;
	    Integer yCountInt = tokens.get(y);
	    if (yCountInt == null) {
	      yCount = 0.0;
	    } else {
	      yCount = (double) yCountInt;
	    }
	    
	    final double emptyCount;
	    Integer emptyCountInt = tokens.get("");
	    if (emptyCountInt == null) {
	    	emptyCount = 0.0;
	    } else {
	    	emptyCount = (double) emptyCountInt;
	    }
	    
	    double pz = (zCount + lambda) / (emptyCount + lambda * vocabSize);
	    double pzy = (yzCount + lambda * vocabSize * pz) / (yCount + lambda * vocabSize);
	    double pzxy = (xyzCount + lambda * vocabSize * pzy) / (xyCount + lambda * vocabSize);
	    return pzxy;
	  }  
}
