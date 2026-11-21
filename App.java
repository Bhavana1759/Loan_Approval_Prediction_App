import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;

public class App extends JFrame {

    private final List<String> featureNames = new ArrayList<>();
    private final List<Boolean> isNumericCol = new ArrayList<>();
    private final List<Map<String, Integer>> colEncoders = new ArrayList<>();
    private final List<List<String>> colCategories = new ArrayList<>();
    private final List<Double> colMeans = new ArrayList<>();
    private final List<String> targetClasses = new ArrayList<>();

    private RandomForest forest;
    private final List<JComponent> inputComponents = new ArrayList<>();
    private JLabel predictionLabel;
    private JLabel probabilityLabel;

    public App() {
        setTitle("Loan Approval Prediction");
        setSize(950, 760);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        File csvFile = new File("credtech_data.csv");
        if (!csvFile.exists()) {
            JOptionPane.showMessageDialog(this,
                    "File 'credtech_data.csv' not found. Please place it in the project folder.",
                    "Error", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
        }

        try {
            loadAndTrain(csvFile);
            buildUI(csvFile);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error loading dataset: " + e.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void loadAndTrain(File file) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath());
        lines.removeIf(String::isBlank);

        String headerLine = lines.get(0);
        String sep = headerLine.contains("\t") ? "\t" : ",";
        String[] headers = headerLine.split(sep, -1);

        int numCols = headers.length;
        int targetIdx = numCols - 1;
        for (int i = 0; i < numCols; i++) {
            headers[i] = headers[i].trim();
            if (headers[i].equalsIgnoreCase("loan_status")) {
                targetIdx = i;
            }
        }

        List<String[]> rawRows = new ArrayList<>();
        for (int r = 1; r < lines.size(); r++) {
            String[] parts = lines.get(r).split(sep, -1);
            if (parts.length == numCols) {
                for (int c = 0; c < numCols; c++) parts[c] = parts[c].trim();
                rawRows.add(parts);
            }
        }

        List<Integer> featIndices = new ArrayList<>();
        for (int c = 0; c < numCols; c++) {
            if (c != targetIdx) {
                featIndices.add(c);
                featureNames.add(headers[c]);
            }
        }

        int numFeatures = featIndices.size();
        for (int f = 0; f < numFeatures; f++) {
            int col = featIndices.get(f);
            boolean numeric = true;
            double sum = 0;
            int count = 0;

            for (String[] row : rawRows) {
                try {
                    double val = Double.parseDouble(row[col]);
                    sum += val;
                    count++;
                } catch (NumberFormatException e) {
                    numeric = false;
                }
            }

            isNumericCol.add(numeric);
            colEncoders.add(new LinkedHashMap<>());
            colCategories.add(new ArrayList<>());
            colMeans.add(numeric && count > 0 ? sum / count : 0.0);

            if (!numeric) {
                Set<String> uniqueVals = new TreeSet<>();
                for (String[] row : rawRows) uniqueVals.add(row[col]);
                int code = 0;
                for (String val : uniqueVals) {
                    colEncoders.get(f).put(val, code++);
                    colCategories.get(f).add(val);
                }
            }
        }

        Set<String> uniqueTargets = new TreeSet<>();
        for (String[] row : rawRows) uniqueTargets.add(row[targetIdx]);
        Map<String, Integer> targetEncoder = new LinkedHashMap<>();
        int tCode = 0;
        for (String t : uniqueTargets) {
            targetEncoder.put(t, tCode++);
            targetClasses.add(t);
        }

        int n = rawRows.size();
        double[][] X = new double[n][numFeatures];
        int[] y = new int[n];

        for (int r = 0; r < n; r++) {
            String[] row = rawRows.get(r);
            for (int f = 0; f < numFeatures; f++) {
                int col = featIndices.get(f);
                if (isNumericCol.get(f)) {
                    X[r][f] = Double.parseDouble(row[col]);
                } else {
                    X[r][f] = colEncoders.get(f).get(row[col]);
                }
            }
            y[r] = targetEncoder.get(row[targetIdx]);
        }

        forest = new RandomForest(50, 8, targetClasses.size(), 42L);
        forest.fit(X, y);
    }

    private void buildUI(File csvFile) throws IOException {
        JPanel root = new JPanel(new BorderLayout(0, 15));
        root.setBackground(new Color(18, 24, 38));
        root.setBorder(new EmptyBorder(25, 40, 25, 40));

        JPanel header = new JPanel();
        header.setLayout(new BoxLayout(header, BoxLayout.Y_AXIS));
        header.setOpaque(false);

        JLabel title = new JLabel("Loan Approval Prediction App");
        title.setForeground(Color.WHITE);
        title.setFont(new Font("SansSerif", Font.BOLD, 26));

        JLabel subtitle = new JLabel("Predict loan approval based on applicant details.");
        subtitle.setForeground(new Color(180, 190, 210));
        subtitle.setFont(new Font("SansSerif", Font.PLAIN, 14));
        subtitle.setBorder(new EmptyBorder(4, 0, 10, 0));

        header.add(title);
        header.add(subtitle);
        root.add(header, BorderLayout.NORTH);

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));
        content.setOpaque(false);

        JLabel previewLbl = new JLabel("Dataset Preview (Top 5 Rows)");
        previewLbl.setForeground(Color.WHITE);
        previewLbl.setFont(new Font("SansSerif", Font.BOLD, 16));
        previewLbl.setAlignmentX(Component.LEFT_ALIGNMENT);
        content.add(previewLbl);
        content.add(Box.createVerticalStrut(8));

        List<String> lines = Files.readAllLines(csvFile.toPath());
        String sep = lines.get(0).contains("\t") ? "\t" : ",";
        String[] cols = lines.get(0).split(sep, -1);
        DefaultTableModel tableModel = new DefaultTableModel(cols, 0);
        for (int i = 1; i <= Math.min(5, lines.size() - 1); i++) {
            tableModel.addRow(lines.get(i).split(sep, -1));
        }

        JTable previewTable = new JTable(tableModel);
        previewTable.setEnabled(false);
        previewTable.setRowHeight(24);
        JScrollPane tableScroll = new JScrollPane(previewTable);
        tableScroll.setPreferredSize(new Dimension(850, 148));
        tableScroll.setMaximumSize(new Dimension(Integer.MAX_VALUE, 148));
        tableScroll.setAlignmentX(Component.LEFT_ALIGNMENT);
        content.add(tableScroll);
        content.add(Box.createVerticalStrut(20));

        JLabel formLbl = new JLabel("Enter Applicant Details");
        formLbl.setForeground(Color.WHITE);
        formLbl.setFont(new Font("SansSerif", Font.BOLD, 16));
        formLbl.setAlignmentX(Component.LEFT_ALIGNMENT);
        content.add(formLbl);
        content.add(Box.createVerticalStrut(10));

        JPanel grid = new JPanel(new GridLayout(0, 2, 24, 14));
        grid.setOpaque(false);
        grid.setAlignmentX(Component.LEFT_ALIGNMENT);

        for (int f = 0; f < featureNames.size(); f++) {
            String name = featureNames.get(f);
            JPanel fieldBox = new JPanel(new BorderLayout(0, 5));
            fieldBox.setOpaque(false);

            if (isNumericCol.get(f)) {
                JLabel lbl = new JLabel("Enter " + name);
                lbl.setForeground(Color.WHITE);
                lbl.setFont(new Font("SansSerif", Font.PLAIN, 13));

                JTextField tf = new JTextField(String.format(Locale.US, "%.2f", colMeans.get(f)));
                tf.setFont(new Font("SansSerif", Font.PLAIN, 14));
                tf.setBorder(new EmptyBorder(6, 8, 6, 8));

                inputComponents.add(tf);
                fieldBox.add(lbl, BorderLayout.NORTH);
                fieldBox.add(tf, BorderLayout.CENTER);
            } else {
                JLabel lbl = new JLabel("Select " + name);
                lbl.setForeground(Color.WHITE);
                lbl.setFont(new Font("SansSerif", Font.PLAIN, 13));

                JComboBox<String> combo = new JComboBox<>(colCategories.get(f).toArray(new String[0]));
                combo.setFont(new Font("SansSerif", Font.PLAIN, 14));

                inputComponents.add(combo);
                fieldBox.add(lbl, BorderLayout.NORTH);
                fieldBox.add(combo, BorderLayout.CENTER);
            }
            grid.add(fieldBox);
        }

        content.add(grid);

        JScrollPane mainScroll = new JScrollPane(content);
        mainScroll.setOpaque(false);
        mainScroll.getViewport().setOpaque(false);
        mainScroll.setBorder(null);
        mainScroll.getVerticalScrollBar().setUnitIncrement(16);
        root.add(mainScroll, BorderLayout.CENTER);

        JPanel footer = new JPanel();
        footer.setLayout(new BoxLayout(footer, BoxLayout.Y_AXIS));
        footer.setOpaque(false);

        JButton predictBtn = new JButton("Predict Loan Status");
        predictBtn.setFont(new Font("SansSerif", Font.BOLD, 15));
        predictBtn.setBackground(new Color(37, 99, 235));
        predictBtn.setForeground(Color.WHITE);
        predictBtn.setFocusPainted(false);
        predictBtn.setBorder(new EmptyBorder(10, 22, 10, 22));
predictBtn.setCursor(new Cursor(Cursor.HAND_CURSOR));
predictBtn.setAlignmentX(Component.LEFT_ALIGNMENT);
predictionLabel = createBanner(new Color(16, 163, 98));
probabilityLabel = createBanner(new Color(30, 115, 190));
predictBtn.addActionListener(e -> handlePredict());
footer.add(predictBtn);
footer.add(Box.createVerticalStrut(12));
footer.add(predictionLabel);
footer.add(Box.createVerticalStrut(8));
footer.add(probabilityLabel);
root.add(footer, BorderLayout.SOUTH);
setContentPane(root);
}
private JLabel createBanner(Color bg) {
JLabel lbl = new JLabel(" ");
lbl.setOpaque(true);
lbl.setBackground(bg);
lbl.setForeground(Color.WHITE);
lbl.setFont(new Font("SansSerif", Font.BOLD, 15));
lbl.setBorder(new EmptyBorder(10, 14, 10, 14));
lbl.setMaximumSize(new Dimension(Integer.MAX_VALUE, 42));
lbl.setAlignmentX(Component.LEFT_ALIGNMENT);
lbl.setVisible(false);
return lbl;
}
private void handlePredict() {
try {
double[] sample = new double[featureNames.size()];
for (int f = 0; f < featureNames.size(); f++) {
if (isNumericCol.get(f)) {
JTextField tf = (JTextField) inputComponents.get(f);
sample[f] = Double.parseDouble(tf.getText().trim());
} else {
@SuppressWarnings("unchecked")
JComboBox combo = (JComboBox) inputComponents.get(f);
String selected = (String) combo.getSelectedItem();
sample[f] = colEncoders.get(f).get(selected);
}
}
double[] proba = forest.predictProba(sample);
int bestClass = 0;
for (int c = 1; c < proba.length; c++) {
if (proba[c] > proba[bestClass]) bestClass = c;
}
String result = targetClasses.get(bestClass);
predictionLabel.setBackground(new Color(16, 163, 98));
predictionLabel.setText("Prediction: " + result);
predictionLabel.setVisible(true);
probabilityLabel.setText(String.format(Locale.US, "Probability: %.2f", proba[bestClass]));
probabilityLabel.setVisible(true);
} catch (NumberFormatException ex) {
predictionLabel.setBackground(new Color(217, 70, 70));
predictionLabel.setText("Please enter valid numeric values.");
predictionLabel.setVisible(true);
probabilityLabel.setVisible(false);
}
}
private static class RandomForest {
private final List trees = new ArrayList<>();
private final int numClasses;
private final Random rng;
RandomForest(int nTrees, int maxDepth, int numClasses, long seed) {
this.numClasses = numClasses;
this.rng = new Random(seed);
for (int i = 0; i < nTrees; i++) {
trees.add(new DecisionTree(maxDepth, numClasses, rng));
}
}
void fit(double[][] X, int[] y) {
int n = X.length;
for (DecisionTree tree : trees) {
double[][] bootX = new double[n][];
int[] bootY = new int[n];
for (int i = 0; i < n; i++) {
int idx = rng.nextInt(n);
bootX[i] = X[idx];
bootY[i] = y[idx];
}
tree.fit(bootX, bootY);
}
}
double[] predictProba(double[] x) {
double[] counts = new double[numClasses];
for (DecisionTree tree : trees) {
counts[tree.predict(x)] += 1.0;
}
for (int c = 0; c < numClasses; c++) {
counts[c] /= trees.size();
}
return counts;
}
}
private static class DecisionTree {
private static class Node {
int feature = -1;
double threshold;
int predictedClass;
Node left, right;
}
private final int maxDepth;
private final int numClasses;
private final Random rng;
private Node root;
DecisionTree(int maxDepth, int numClasses, Random rng) {
this.maxDepth = maxDepth;
this.numClasses = numClasses;
this.rng = rng;
}
void fit(double[][] X, int[] y) {
List indices = new ArrayList<>();
for (int i = 0; i < X.length; i++) indices.add(i);
root = build(X, y, indices, 0);
}
private Node build(double[][] X, int[] y, List idx, int depth) {
Node node = new Node();
int[] classCounts = new int[numClasses];
for (int i : idx) classCounts[y[i]]++;
int majority = 0;
for (int c = 1; c < numClasses; c++) {
if (classCounts[c] > classCounts[majority]) majority = c;
}
node.predictedClass = majority;
if (depth >= maxDepth || idx.size() <= 4 || classCounts[majority] == idx.size()) {
return node;
}
int numFeatures = X[0].length;
int tryFeatures = Math.max(1, (int) Math.sqrt(numFeatures));
double bestGini = Double.MAX_VALUE;
int bestFeat = -1;
double bestThresh = 0;
List bestLeft = null, bestRight = null;
for (int fTry = 0; fTry < tryFeatures; fTry++) {
int f = rng.nextInt(numFeatures);
double thresh = X[idx.get(rng.nextInt(idx.size()))][f];
List left = new ArrayList<>();
List right = new ArrayList<>();
for (int i : idx) {
if (X[i][f] <= thresh) left.add(i);
else right.add(i);
}
if (left.isEmpty() || right.isEmpty()) continue;
double gini = (left.size() * gini(y, left) + right.size() * gini(y, right)) / idx.size();
if (gini < bestGini) {
bestGini = gini;
bestFeat = f;
bestThresh = thresh;
bestLeft = left;
bestRight = right;
}
}
if (bestFeat == -1) return node;
node.feature = bestFeat;
node.threshold = bestThresh;
node.left = build(X, y, bestLeft, depth + 1);
node.right = build(X, y, bestRight, depth + 1);
return node;
}
private double gini(int[] y, List idx) {
int[] counts = new int[numClasses];
for (int i : idx) counts[y[i]]++;
double impurity = 1.0;
for (int c : counts) {
double p = (double) c / idx.size();
impurity -= p * p;
}
return impurity;
}
int predict(double[] x) {
Node curr = root;
while (curr.feature != -1) {
curr = (x[curr.feature] <= curr.threshold) ? curr.left : curr.right;
}
return curr.predictedClass;
}
}
public static void main(String[] args) {
SwingUtilities.invokeLater(() -> new App().setVisible(true));
}
}
