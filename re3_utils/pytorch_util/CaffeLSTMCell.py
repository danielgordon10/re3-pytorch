import torch
import torch.nn as nn
import torch.nn.functional as F


class CaffeLSTMCell(nn.Module):
    def __init__(self, input_size, output_size):
        super(CaffeLSTMCell, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.block_input = nn.Linear(input_size + output_size, output_size)
        self.input_gate = nn.Linear(input_size + output_size * 2, output_size)
        self.forget_gate = nn.Linear(input_size + output_size * 2, output_size)
        self.output_gate = nn.Linear(input_size + output_size * 2, output_size)

    def forward(self, inputs, hx=None):
        if hx is None or (hx[0] is None and hx[1] is None):
            zeros = torch.zeros(inputs.size(0), self.output_size, dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)

        cell_outputs_prev, cell_state_prev = hx

        lstm_concat = torch.cat([inputs, cell_outputs_prev], 1)
        peephole_concat = torch.cat([lstm_concat, cell_state_prev], 1)

        block_input = torch.tanh(self.block_input(lstm_concat))

        input_gate = torch.sigmoid(self.input_gate(peephole_concat))
        input_mult = input_gate * block_input

        forget_gate = torch.sigmoid(self.forget_gate(peephole_concat))
        forget_mult = forget_gate * cell_state_prev

        cell_state_new = input_mult + forget_mult
        cell_state_activated = torch.tanh(cell_state_new)

        output_concat = torch.cat([lstm_concat, cell_state_new], 1)
        output_gate = torch.sigmoid(self.output_gate(output_concat))
        cell_outputs_new = output_gate * cell_state_activated

        return cell_outputs_new, cell_state_new
