import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.DataInput;
import java.io.DataOutput;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SVM {

    // Mapper class
    public static class SVMMapper extends Mapper<LongWritable, Text, Text, DoubleWritableArray> {
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Parse input line and extract features and label
            String[] tokens = value.toString().split("\\s+"); // Assuming space-separated values
            double label = Double.parseDouble(tokens[tokens.length - 1]); // Assuming label is the last token
            double[] features = new double[tokens.length - 1];
            for (int i = 0; i < tokens.length - 1; i++) {
                features[i] = Double.parseDouble(tokens[i]);
            }
            // Emit feature array and label
            context.write(new Text("data"), new DoubleWritableArray(features, label));
        }
    }

    // Reducer class
    public static class SVMReducer extends Reducer<Text, DoubleWritableArray, Text, DoubleWritable> {
        private static final double LEARNING_RATE = 0.01;
        private static final int MAX_ITERATIONS = 100;
        private static final double TOLERANCE = 0.001;

        @Override
        public void reduce(Text key, Iterable<DoubleWritableArray> values, Context context)
                throws IOException, InterruptedException {
            // Gather features and labels
            List<double[]> featureList = new ArrayList<>();
            List<Double> labelList = new ArrayList<>();
            for (DoubleWritableArray val : values) {
                featureList.add(val.getArray());
                labelList.add(val.getLabel());
            }

            // Train SVM using SMO algorithm
            int numFeatures = featureList.get(0).length;
            double[] weights = new double[numFeatures];
            // Initialize weights to zeros
            for (int i = 0; i < numFeatures; i++) {
                weights[i] = 0;
            }
            double bias = 0;
            int numChanged = 0;
            int iter = 0;
            while (iter < MAX_ITERATIONS && numChanged > 0) {
                numChanged = 0;
                for (int i = 0; i < featureList.size(); i++) {
                    double[] features = featureList.get(i);
                    double label = labelList.get(i);
                    double error = label - classify(weights, bias, features);
                    if ((label * error < -TOLERANCE && weights[i] < context.getConfiguration().getDouble("C", 1.0))
                            || (label * error > TOLERANCE && weights[i] > 0)) {
                        for (int j = 0; j < numFeatures; j++) {
                            weights[j] += LEARNING_RATE * (label - classify(weights, bias, features)) * features[j];
                        }
                        bias += LEARNING_RATE * label;
                        numChanged++;
                    }
                }
                iter++;
            }

            // Output weights and bias
            for (int i = 0; i < numFeatures; i++) {
                context.write(new Text("weight" + i), new DoubleWritable(weights[i]));
            }
            context.write(new Text("bias"), new DoubleWritable(bias));
        }

        private double classify(double[] weights, double bias, double[] features) {
            double result = 0;
            for (int i = 0; i < weights.length; i++) {
                result += weights[i] * features[i];
            }
            result += bias;
            return result;
        }
    }

    // Driver code
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: SVM <input path> <output path> <C>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        conf.setDouble("C", Double.parseDouble(args[2]));

        Job job = Job.getInstance(conf, "SVM");

        job.setJarByClass(SVM.class);
        job.setMapperClass(SVMMapper.class);
        job.setReducerClass(SVMReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    // Custom Writable class for double array
    public static class DoubleWritableArray implements Writable {
        private double[] array;
        private double label;

        public DoubleWritableArray() {
            // Default constructor required for Hadoop serialization
        }

        public DoubleWritableArray(double[] array, double label) {
            this.array = array;
            this.label = label;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeInt(array.length);
            for (double value : array) {
                out.writeDouble(value);
            }
            out.writeDouble(label);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            int length = in.readInt();
            array = new double[length];
            for (int i = 0; i < length; i++) {
                array[i] = in.readDouble();
            }
            label = in.readDouble();
        }

        public double[] getArray() {
            return array;
        }

        public double getLabel() {
            return label;
        }
    }
}
