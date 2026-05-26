import tkinter as tk
import math
import random

# --- RBM Core Logic ---
class RestrictedBoltzmannMachine:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # Initialize weights with small random values
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(num_hidden)] for _ in range(num_visible)]
        self.v_biases = [0.0] * num_visible
        self.h_biases = [0.0] * num_hidden
        
        # Node states: probabilities (continuous) and sampled states (0 or 1)
        self.v_probs = [0.5] * num_visible
        self.v_states = [0] * num_visible
        self.h_probs = [0.5] * num_hidden
        self.h_states = [0] * num_hidden
        
        # Track which layer was updated last (for step-by-step demo)
        self.last_updated_layer = "visible" 

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(min(x, 50), -50))) # Prevents overflow

    def sample_hidden(self):
        """Compute hidden probabilities and sample states given visible states."""
        for j in range(self.num_hidden):
            activation = self.h_biases[j]
            for i in range(self.num_visible):
                activation += self.v_states[i] * self.weights[i][j]
            self.h_probs[j] = self.sigmoid(activation)
            self.h_states[j] = 1 if random.random() < self.h_probs[j] else 0
        self.last_updated_layer = "hidden"

    def sample_visible(self):
        """Compute visible probabilities and sample states given hidden states."""
        for i in range(self.num_visible):
            activation = self.v_biases[i]
            for j in range(self.num_hidden):
                activation += self.h_states[j] * self.weights[i][j]
            self.v_probs[i] = self.sigmoid(activation)
            self.v_states[i] = 1 if random.random() < self.v_probs[i] else 0
        self.last_updated_layer = "visible"

    def contrastive_divergence_1(self, target_pattern, lr=0.2):
        """Perform one step of Contrastive Divergence (CD-1) learning."""
        # 1. Positive phase: Set visible to target, sample hidden
        v0_states = list(target_pattern)
        h0_probs = [0.0] * self.num_hidden
        h0_states = [0] * self.num_hidden
        
        for j in range(self.num_hidden):
            activation = self.h_biases[j]
            for i in range(self.num_visible):
                activation += v0_states[i] * self.weights[i][j]
            h0_probs[j] = self.sigmoid(activation)
            h0_states[j] = 1 if random.random() < h0_probs[j] else 0

        # 2. Negative phase: Reconstruction (v1), then sample h1 again
        v1_probs = [0.0] * self.num_visible
        v1_states = [0] * self.num_visible
        for i in range(self.num_visible):
            activation = self.v_biases[i]
            for j in range(self.num_hidden):
                activation += h0_states[j] * self.weights[i][j]
            v1_probs[i] = self.sigmoid(activation)
            v1_states[i] = 1 if random.random() < v1_probs[i] else 0

        h1_probs = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            activation = self.h_biases[j]
            for i in range(self.num_visible):
                activation += v1_states[i] * self.weights[i][j]
            h1_probs[j] = self.sigmoid(activation)

        # 3. Update weights and biases
        for i in range(self.num_visible):
            for j in range(self.num_hidden):
                self.weights[i][j] += lr * (v0_states[i] * h0_probs[j] - v1_states[i] * h1_probs[j])
                
        for i in range(self.num_visible):
            self.v_biases[i] += lr * (v0_states[i] - v1_states[i])
            
        for j in range(self.num_hidden):
            self.h_biases[j] += lr * (h0_probs[j] - h1_probs[j])

        # Sync the GUI state with the positive phase result for display
        self.v_states = v0_states
        self.h_probs = h0_probs
        self.h_states = h0_states
        self.last_updated_layer = "hidden"


# --- GUI Application ---
class RBMVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Restricted Boltzmann Machine (RBM) Interactive Demo")
        self.geometry("850x650")
        self.resizable(False, False)

        # Initialize RBM (4 visible nodes, 3 hidden nodes for clear visualization)
        self.rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)
        
        # Target pattern for training (Alternating pattern as default)
        self.training_pattern = [1, 0, 1, 0]

        self.setup_widgets()
        self.draw_network()

    def setup_widgets(self):
        # Left Panel: Controls
        control_frame = tk.Frame(self, width=250, padx=10, pady=10, relief=tk.RIDGE, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="RBM Controls", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Target Pattern Setup
        tk.Label(control_frame, text="Target Pattern for Training:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.pattern_vars = []
        pattern_frame = tk.Frame(control_frame)
        pattern_frame.pack(anchor=tk.W, pady=5)
        for i in range(self.rbm.num_visible):
            var = tk.IntVar(value=self.training_pattern[i])
            cb = tk.Checkbutton(pattern_frame, text=f"V{i}", variable=var, command=self.update_target_pattern)
            cb.pack(side=tk.LEFT, padx=2)
            self.pattern_vars.append(var)

        tk.Frame(control_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)

        # Sampling Actions
        tk.Label(control_frame, text="Gibbs Sampling", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        tk.Button(control_frame, text="Step Forward (Alternating)", command=self.step_forward, bg="#e1f5fe", height=2, width=25).pack(pady=5)
        tk.Button(control_frame, text="Randomize States", command=self.randomize_states, width=25).pack(pady=5)

        tk.Frame(control_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=15)

        # Training Actions
        tk.Label(control_frame, text="Learning (CD-1)", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        tk.Button(control_frame, text="Train 1 Step", command=self.train_1_step, bg="#e8f5e9", width=25).pack(pady=5)
        tk.Button(control_frame, text="Train 100 Steps", command=self.train_100_steps, bg="#c8e6c9", width=25).pack(pady=5)
        tk.Button(control_frame, text="Reset Weights & Biases", command=self.reset_parameters, fg="red", width=25).pack(pady=20)

        # Info Box
        self.status_text = tk.StringVar(value="Status: Ready\nClick 'Step Forward' to sample.")
        status_label = tk.Label(control_frame, textvariable=self.status_text, justify=tk.LEFT, font=("Courier", 9), bg="#f5f5f5", relief=tk.SUNKEN, bd=1, height=6, width=28)
        status_label.pack(side=tk.BOTTOM, pady=10)

        # Right Panel: Canvas for network visualization
        self.canvas = tk.Canvas(self, bg="white", width=600, height=650)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def update_target_pattern(self):
        self.training_pattern = [var.get() for var in self.pattern_vars]
        self.status_text.set(f"Updated training target\nto: {self.training_pattern}")

    def randomize_states(self):
        self.rbm.v_states = [random.choice([0, 1]) for _ in range(self.rbm.num_visible)]
        self.rbm.h_states = [random.choice([0, 1]) for _ in range(self.rbm.num_hidden)]
        self.status_text.set("Node states randomized randomly.")
        self.draw_network()

    def reset_parameters(self):
        self.rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)
        self.update_target_pattern()
        self.status_text.set("Weights and biases reset\nto small random values.")
        self.draw_network()

    def step_forward(self):
        if self.rbm.last_updated_layer == "visible":
            self.rbm.sample_hidden()
            self.status_text.set("Sampled Hidden layer\nbased on Visible layer.")
        else:
            self.rbm.sample_visible()
            self.status_text.set("Sampled Visible layer\nbased on Hidden layer.")
        self.draw_network()

    def train_1_step(self):
        self.rbm.contrastive_divergence_1(self.training_pattern)
        self.status_text.set(f"Trained 1 step with CD-1\nTarget: {self.training_pattern}")
        self.draw_network()

    def train_100_steps(self):
        for _ in range(100):
            self.rbm.contrastive_divergence_1(self.training_pattern)
        self.status_text.set(f"Trained 100 steps with CD-1\nTarget: {self.training_pattern}")
        self.draw_network()

    def draw_network(self):
        self.canvas.delete("all")

        # Layout dimensions
        v_x = 150
        h_x = 450
        canvas_height = 600
        
        v_positions = []
        h_positions = []

        # Title / Legend
        self.canvas.create_text(300, 30, text="Restricted Boltzmann Machine Topology", font=("Helvetica", 16, "bold"), fill="#333333")
        self.canvas.create_text(300, 55, text="Line thickness = Weight absolute value  |  Blue = Positive, Red = Negative", font=("Helvetica", 9, "italic"), fill="#666666")

        # Calculate Y positions for layers
        for i in range(self.rbm.num_visible):
            y = (canvas_height / (self.rbm.num_visible + 1)) * (i + 1) + 20
            v_positions.append((v_x, y))

        for j in range(self.rbm.num_hidden):
            y = (canvas_height / (self.rbm.num_hidden + 1)) * (j + 1) + 20
            h_positions.append((h_x, y))

        # 1. Draw Weights (Edges)
        for i in range(self.rbm.num_visible):
            for j in range(self.rbm.num_hidden):
                w = self.rbm.weights[i][j]
                color = "#2196F3" if w >= 0 else "#F44336"  # Blue for positive, Red for negative
                thickness = max(1, min(int(abs(w) * 5), 8)) # Scale thickness to weight magnitude
                self.canvas.create_line(v_positions[i][0], v_positions[i][1], h_positions[j][0], h_positions[j][1], fill=color, width=thickness)
                
                # Draw weight numeric values at the middle of edges
                mid_x = (v_positions[i][0] + h_positions[j][0]) / 2
                mid_y = (v_positions[i][1] + h_positions[j][1]) / 2
                # Offset slightly to avoid overlap
                offset = (j - 1) * 12
                self.canvas.create_text(mid_x, mid_y + offset, text=f"{w:.2f}", font=("Helvetica", 8), fill="#555555")

        # 2. Draw Visible Nodes
        for i, (x, y) in enumerate(v_positions):
            state = self.rbm.v_states[i]
            prob = self.rbm.v_probs[i]
            bias = self.rbm.v_biases[i]
            
            # Node color based on binary sampled state (1=Yellow active, 0=Gray inactive)
            node_color = "#FFD54F" if state == 1 else "#E0E0E0"
            outline_color = "#FF8F00" if self.rbm.last_updated_layer == "visible" else "#9E9E9E"
            outline_width = 3 if self.rbm.last_updated_layer == "visible" else 1

            self.canvas.create_oval(x-25, y-25, x+25, y+25, fill=node_color, outline=outline_color, width=outline_width)
            self.canvas.create_text(x, y-5, text=f"V{i}\n[{state}]", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(x, y+15, text=f"p={prob:.2f}", font=("Helvetica", 8))
            self.canvas.create_text(x-50, y, text=f"b={bias:.2f}", font=("Helvetica", 9), anchor=tk.E)

        # 3. Draw Hidden Nodes
        for j, (x, y) in enumerate(h_positions):
            state = self.rbm.h_states[j]
            prob = self.rbm.h_probs[j]
            bias = self.rbm.h_biases[j]
            
            node_color = "#FFD54F" if state == 1 else "#E0E0E0"
            outline_color = "#FF8F00" if self.rbm.last_updated_layer == "hidden" else "#9E9E9E"
            outline_width = 3 if self.rbm.last_updated_layer == "hidden" else 1

            self.canvas.create_oval(x-25, y-25, x+25, y+25, fill=node_color, outline=outline_color, width=outline_width)
            self.canvas.create_text(x, y-5, text=f"H{j}\n[{state}]", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(x, y+15, text=f"p={prob:.2f}", font=("Helvetica", 8))
            self.canvas.create_text(x+50, y, text=f"b={bias:.2f}", font=("Helvetica", 9), anchor=tk.W)

        # Layer Labels
        self.canvas.create_text(v_x, canvas_height - 10, text="Visible Layer (v)", font=("Helvetica", 12, "bold"))
        self.canvas.create_text(h_x, canvas_height - 10, text="Hidden Layer (h)", font=("Helvetica", 12, "bold"))


if __name__ == "__main__":
    # Ensure reproducibility for initial state demonstration
    random.seed(42)
    app = RBMVisualizer()
    app.mainloop()