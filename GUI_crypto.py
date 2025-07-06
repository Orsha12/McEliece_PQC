import tkinter as tk
from tkinter import ttk
import numpy as np
import galois
from core import LinearCode, EXAMPLE_1, create_error_vector

GF2 = galois.GF(2)

class McElieceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔐 Linear Code Simulator")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f4f8")

        A, S, P = EXAMPLE_1
        self.code = LinearCode(A, S, P)

        self.message = None
        self.error = None
        self.plaintext = None

        self.build_style()
        self.build_layout()

    def build_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#f0f4f8", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.configure("TEntry", font=("Consolas", 11))
        style.configure("TLabelframe", font=("Segoe UI", 12, "bold"), background="#f0f4f8")
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"), foreground="#005c99")

    def build_layout(self):
        # Probability Inputs
        prob_frame = ttk.LabelFrame(self.root, text="🎯 Error Probabilities", padding=15)
        prob_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(prob_frame, text="Decoder Error Probability (0–1):").grid(row=0, column=0, sticky="w", pady=5)
        self.decoder_p_entry = ttk.Entry(prob_frame, width=10)
        self.decoder_p_entry.insert(0, "0.03")
        self.decoder_p_entry.grid(row=0, column=1, padx=10)

        ttk.Label(prob_frame, text="Channel Error Probability (0–1):").grid(row=1, column=0, sticky="w", pady=5)
        self.channel_p_entry = ttk.Entry(prob_frame, width=10)
        self.channel_p_entry.insert(0, "0.02")
        self.channel_p_entry.grid(row=1, column=1, padx=10)

        ttk.Label(prob_frame, text="Number of Runs:").grid(row=2, column=0, sticky="w", pady=5)
        self.runs_entry = ttk.Entry(prob_frame, width=10)
        self.runs_entry.insert(0, "50")
        self.runs_entry.grid(row=2, column=1, padx=10)

        ttk.Button(prob_frame, text="📈 Run Channel Simulation", command=self.run_simulation).grid(row=3, column=0, columnspan=2, pady=10)

        # Input Stage
        input_frame = ttk.LabelFrame(self.root, text="🧾Input Stage", padding=15)
        input_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(input_frame, text=f"Message length (k): {self.code.k}").grid(row=0, column=0, sticky="w")
        ttk.Label(input_frame, text=f"Error vector length (n): {self.code.n}, max 1's: {self.code.max_errors_num}").grid(row=1, column=0, sticky="w", pady=5)

        ttk.Label(input_frame, text="Binary Message:").grid(row=2, column=0, sticky="w", pady=5)
        self.message_entry = ttk.Entry(input_frame, width=70)
        self.message_entry.grid(row=2, column=1, padx=10)

        ttk.Label(input_frame, text="Error Vector:").grid(row=3, column=0, sticky="w", pady=5)
        self.error_entry = ttk.Entry(input_frame, width=70)
        self.error_entry.grid(row=3, column=1, padx=10)

        ttk.Button(input_frame, text="🎲 Generate Random Inputs", command=self.generate_random_inputs).grid(row=4, column=0, columnspan=2, pady=10)

        # Actions
        action_frame = ttk.LabelFrame(self.root, text="🛠️ Actions", padding=15)
        action_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(action_frame, text="🔒 Encode", command=self.encode).pack(side="left", padx=10)
        ttk.Button(action_frame, text="🔓 Decode", command=self.decode).pack(side="left", padx=10)
        ttk.Button(action_frame, text="🧹 Clear Results", command=self.clear_all).pack(side="left", padx=10)

        # Results
        result_frame = ttk.LabelFrame(self.root, text="📋 Results", padding=15)
        result_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.output_text = tk.Text(result_frame, wrap="word", font=("Consolas", 11), bg="#ffffff", fg="#333333")
        self.output_text.pack(fill="both", expand=True)

    def generate_random_inputs(self):
        self.message = GF2.Random(self.code.k)
        self.error = GF2(create_error_vector(self.code.n, self.code.max_errors_num))

        self.message_entry.delete(0, tk.END)
        self.message_entry.insert(0, ''.join(map(str, self.message.tolist())))

        self.error_entry.delete(0, tk.END)
        self.error_entry.insert(0, ''.join(map(str, self.error.tolist())))

        self.output_text.insert(tk.END, "🎲 Random inputs generated.\n")

    def encode(self):
        self.output_text.delete("1.0", tk.END)
        try:
            msg_str = self.message_entry.get().strip().replace(" ", "")
            err_str = self.error_entry.get().strip().replace(" ", "")

            if len(msg_str) != self.code.k:
                raise ValueError(f"Message must be exactly {self.code.k} bits.")
            if len(err_str) != self.code.n:
                raise ValueError(f"Error must be exactly {self.code.n} bits.")
            if err_str.count("1") > self.code.max_errors_num:
                raise ValueError(f"Error vector can contain up to {self.code.max_errors_num} ones.")

            self.message = GF2([int(b) for b in msg_str])
            self.error = GF2([int(b) for b in err_str])
            self.plaintext = self.code.encode(self.message, self.error)

            self.output_text.insert(tk.END, f" 🔒 Encoding complete!\n")
            self.output_text.insert(tk.END, f"Message vector:  {self.message}\n")
            self.output_text.insert(tk.END, f"Error vector:    {self.error}\n")
            self.output_text.insert(tk.END, f"Ciphertext:      {self.plaintext}\n")

        except Exception as e:
            self.output_text.insert(tk.END, f"❗ Encoding error: {str(e)}\n")

    def decode(self):
        self.output_text.insert(tk.END, "\n🔓 Decoding...\n")
        try:
            if self.plaintext is None:
                raise ValueError("You must run Encode first.")

            expected_estimated_error = self.error @ self.code.P.T
            expected_syndrome = expected_estimated_error @ self.code.H.T
            decoded_message = self.code.decode(self.plaintext, expected_estimated_error, expected_syndrome)

            self.output_text.insert(tk.END, f"Syndrome:         {expected_syndrome}\n")
            self.output_text.insert(tk.END, f"Decoded message:  {decoded_message}\n")

            if np.array_equal(decoded_message, self.message):
                self.output_text.insert(tk.END, f"✅ Success! Message correctly decoded.\n")
            else:
                self.output_text.insert(tk.END, f"❌ Decoding failed. Message mismatch.\n")

        except Exception as e:
            self.output_text.insert(tk.END, f"❗ Decoding error: {str(e)}\n")

    def clear_all(self):
        self.message_entry.delete(0, tk.END)
        self.error_entry.delete(0, tk.END)
        self.output_text.delete("1.0", tk.END)
        self.message = None
        self.error = None
        self.plaintext = None


# Run the GUI

    def run_simulation(self):
        try:
            decoder_p = float(self.decoder_p_entry.get())
            channel_p = float(self.channel_p_entry.get())
            runs = int(self.runs_entry.get())

            msg_str = self.message_entry.get().strip().replace(" ", "")
            use_message = msg_str if len(msg_str) == self.code.k and all(c in "01" for c in msg_str) else None

            if use_message:
                self.output_text.insert(tk.END, f"📨 Using user message: {msg_str}\n")
            else:
                self.output_text.insert(tk.END, f"📨 No valid user message found. Running with random messages.\n")

            success_count = 0
            total_error_sum = 0
            for i in range(runs):
                success, error_count = self.code.simulate_random_channel(decoder_p, channel_p, use_message)
                success_count += success
                total_error_sum += error_count

            success_rate = success_count / runs * 100
            avg_errors = total_error_sum / runs

            self.output_text.insert(tk.END, f"✅ Success Rate over {runs} runs: {success_rate:.2f}%\n")
            self.output_text.insert(tk.END, f"📊 Average total errors per run: {avg_errors:.2f}\n\n")

        except Exception as e:
            self.output_text.insert(tk.END, f"❗ Simulation error: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = McElieceGUI(root)
    root.mainloop()
