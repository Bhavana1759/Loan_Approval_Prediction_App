import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class TestApp extends JFrame {

    public TestApp() {
        setTitle("Java Swing Test");
        setSize(520, 260);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBackground(new Color(18, 24, 38));
        panel.setBorder(new EmptyBorder(40, 40, 40, 40));

        JLabel title = new JLabel("Hello Java 👋");
        title.setForeground(Color.WHITE);
        title.setFont(new Font("SansSerif", Font.BOLD, 28));

        JLabel message = new JLabel("If you see this, Java Swing is working!");
        message.setForeground(new Color(180, 190, 210));
        message.setFont(new Font("SansSerif", Font.PLAIN, 16));
        message.setBorder(new EmptyBorder(15, 0, 0, 0));

        panel.add(title);
        panel.add(message);
        setContentPane(panel);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new TestApp().setVisible(true));
    }
}
