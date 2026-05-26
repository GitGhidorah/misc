import tkinter as tk
import random

# --- Hopfield Network Core Logic ---
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size  # Total number of neurons (e.g., 25 for a 5x5 grid)
        # Initialize weights to 0. Shape: (size, size)
        self.weights = [[0.0 for _ in range(size)] for _ in range(size)]
        # Neuron states: represented as +1 (active/black) or -1 (inactive/white)
        self.states = [-1] * size

    def train_hebbian(self, pattern):
        """Memorize a pattern using Hebbian learning rule (Outer product)."""
        # pattern is a list of +1 and -1 of length self.size
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    self.weights[i][j] = 0.0  # No self-connections
                else:
                    # Incremental Hebbian learning: W_ij += x_i * x_j
                    self.weights[i][j] += pattern[i] * pattern[j]

    def update_single_neuron(self):
        """Pick a random neuron and update its state asynchronously."""
        idx = random.randint(0, self.size - 1)
        
        # Calculate net input: h_i = sum(W_ij * s_j)
        net_input = 0.0
        for j in range(self.size):
            net_input += self.weights[idx][j] * self.states[j]
            
        # Threshold function
        old_state = self.states[idx]
        if net_input >= 0:
            self.states[idx] = 1
        else:
            self.states[idx] = -1
            
        # Return True if the state actually changed, or the index for tracking
        return idx, (old_state != self.states[idx])

    def calculate_energy(self):
        """Calculate the current Network Energy (Lyapunov function)."""
        # E = -0.5 * sum(W_ij * s_i * s_j)
        energy = 0.0
        for i in range(self.size):
            for j in range(self.size):
                energy += self.weights[i][j] * self.states[i] * self.states[j]
        return -0.5 * energy


# --- GUI Application ---
class HopfieldVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hopfield Network Associative Memory Demo")
        self.geometry("800x550")
        self.resizable(False, False)

        self.grid_side = 5
        self.num_neurons = self.grid_side * self.grid_side
        self.hopfield = HopfieldNetwork(self.num_neurons)

        # Preset Patterns (5x5 shapes)
        # Pattern 'T'
        self.pattern_T = [
             1,  1,  1,  1,  1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1,
            -1, -1,  1, -1, -1
        ]
        # Pattern 'X'
        self.pattern_X = [
             1, -1, -1, -1,  1,
            -1,  1, -1,  1, -1,
            -1, -1,  1, -1, -1,
            -1,  1, -1,  1, -1,
             1, -1, -1, -1,  1
        ]

        self.setup_widgets()
        self.sync_gui_from_network()

    def setup_widgets(self):
        # Left Panel: Controls
        control_frame = tk.Frame(self, width=300, padx=15, pady=15, relief=tk.RIDGE, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Hopfield Controls", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Presets
        tk.Label(control_frame, text="1. Load Preset Shapes:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Shape 'T'", command=lambda: self.load_pattern(self.pattern_T), width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Shape 'X'", command=lambda: self.load_pattern(self.pattern_X), width=10).pack(side=tk.LEFT, padx=2)

        # Learning
        tk.Label(control_frame, text="2. Training:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(15, 2))
        tk.Button(control_frame, text="Memorize Current Pattern", command=self.memorize_current, bg="#e8f5e9", width=26).pack(pady=5)

        tk.Frame(control_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

        # Distort / Noise
        tk.Label(control_frame, text="3. Inject Noise:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        tk.Button(control_frame, text="Flip 3 Random Pixels", command=self.inject_noise, bg="#fff3e0", width=26).pack(pady=5)

        # Reconstruction / Dynamics
        tk.Label(control_frame, text="4. State Dynamics:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(15, 2))
        tk.Button(control_frame, text="Step Update (1 Neuron)", command=self.step_update, bg="#e1f5fe", width=26).pack(pady=5)
        tk.Button(control_frame, text="Run to Convergence", command=self.run_to_convergence, bg="#b3e5fc", width=26).pack(pady=5)
        
        tk.Button(control_frame, text="Clear Network Weights", command=self.clear_weights, fg="red", width=26).pack(pady=15)

        # Status / Info Box
        self.status_text = tk.StringVar(value="Status: Ready\nClick cells to draw,\nthen 'Memorize'.")
        status_label = tk.Label(control_frame, textvariable=self.status_text, justify=tk.LEFT, font=("Courier", 9), bg="#f5f5f5", relief=tk.SUNKEN, bd=1, height=6, width=30)
        status_label.pack(side=tk.BOTTOM, pady=5)

        # Right Panel: Canvas for Grid
        self.canvas_size = 450
        self.cell_size = self.canvas_size // self.grid_side
        self.canvas = tk.Canvas(self, bg="#eeeeee", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.RIGHT, padx=20, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_canvas_click(self, event):
        """Toggle neuron state (+1 / -1) when clicking a canvas cell."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= col < self.grid_side and 0 <= row < self.grid_side:
            idx = row * self.grid_side + col
            # Toggle between 1 and -1
            self.hopfield.states[idx] = -1 if self.hopfield.states[idx] == 1 else 1
            self.draw_grid()
            self.update_status("Toggled pixel manually.")

    def load_pattern(self, pattern):
        self.hopfield.states = list(pattern)
        self.draw_grid()
        self.update_status("Loaded preset shape.")

    def memorize_current(self):
        self.hopfield.train_hebbian(self.hopfield.states)
        self.update_status("Memorized current pattern\nvia Hebbian learning.")

    def inject_noise(self):
        indices = random.sample(range(self.num_neurons), 3)
        for idx in indices:
            self.hopfield.states[idx] = -1 if self.hopfield.states[idx] == 1 else 1
        self.draw_grid()
        self.update_status("Injected noise into\n3 random pixels.")

    def step_update(self):
        idx, changed = self.hopfield.update_single_neuron()
        self.draw_grid(highlight_idx=idx)
        change_str = "Changed!" if changed else "No change."
        self.update_status(f"Updated Neuron {idx}.\nResult: {change_str}")

    def run_to_convergence(self):
        # Run asynchronous updates many times to guarantee local minimum energy
        max_iterations = self.num_neurons * 5
        changes_detected = 0
        
        for _ in range(max_iterations):
            _, changed = self.hopfield.update_single_neuron()
            if changed:
                changes_detected += 1
                
        self.draw_grid()
        self.update_status(f"Converged.\nTotal internal state\nflips: {changes_detected}")

    def clear_weights(self):
        self.hopfield = HopfieldNetwork(self.num_neurons)
        self.sync_gui_from_network()
        self.update_status("Cleared all weights.\nNetwork forgot everything.")

    def sync_gui_from_network(self):
        self.draw_grid()

    def update_status(self, message):
        energy = self.hopfield.calculate_energy()
        self.status_text.set(f"Energy: {energy:.1f}\n\n{message}")

    def draw_grid(self, highlight_idx=None):
        self.canvas.delete("all")
        for i in range(self.num_neurons):
            row = i // self.grid_side
            col = i % self.grid_side
            
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            
            # State 1 = Black, State -1 = White
            fill_color = "#333333" if self.hopfield.states[i] == 1 else "#FFFFFF"
            outline_color = "#999999"
            outline_width = 1
            
            # Highlight the single neuron that was just checked
            if highlight_idx == i:
                outline_color = "#FF1744"
                outline_width = 3
                
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=outline_color, width=outline_width)


if __name__ == "__main__":
    app = HopfieldVisualizer()
    app.mainloop()