#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0
# GNU Radio Python Flow Graph
# Title: Top Block (4 UEs + eNB)
# GNU Radio version: 3.9.5.0

from gnuradio import blocks
from gnuradio import gr
import sys
import signal
from gnuradio import zeromq

class top_block(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Top Block (4 UEs + eNB)")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 23.04e6

        ##################################################
        # Blocks
        ##################################################

        # Request sources (receiving from UEs)
        self.zeromq_req_source_ue1 = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2001', 100, False, -1)
        self.zeromq_req_source_ue2 = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2011', 100, False, -1)
        self.zeromq_req_source_ue3 = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2021', 100, False, -1)
        self.zeromq_req_source_ue4 = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2031', 100, False, -1)

        # eNB request source (receiving from eNB)
        self.zeromq_req_source_enb = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2101', 100, False, -1)

        # Reply sinks (sending to UEs)
        self.zeromq_rep_sink_ue1 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2000', 100, False, -1)
        self.zeromq_rep_sink_ue2 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2010', 100, False, -1)
        self.zeromq_rep_sink_ue3 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2020', 100, False, -1)
        self.zeromq_rep_sink_ue4 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2030', 100, False, -1)

        # eNB reply sink (sending to eNB)
        self.zeromq_rep_sink_enb = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2100', 100, False, -1)

        # Throttle blocks to control sample rate
        self.blocks_throttle_ue1 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_throttle_ue2 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_throttle_ue3 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_throttle_ue4 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)
        self.blocks_throttle_enb = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate, True)

        # Signal combiner (adds up signals from all 4 UEs)
        self.blocks_add_xx = blocks.add_vcc(1)

        ##################################################
        # Connections
        ##################################################
        # UEs data sources -> combiner
        self.connect((self.zeromq_req_source_ue1, 0), (self.blocks_add_xx, 0))
        self.connect((self.zeromq_req_source_ue2, 0), (self.blocks_add_xx, 1))
        self.connect((self.zeromq_req_source_ue3, 0), (self.blocks_add_xx, 2))
        self.connect((self.zeromq_req_source_ue4, 0), (self.blocks_add_xx, 3))

        # Combined signal -> eNB throttle
        self.connect((self.blocks_add_xx, 0), (self.blocks_throttle_enb, 0))

        # eNB throttle -> eNB sink
        self.connect((self.blocks_throttle_enb, 0), (self.zeromq_rep_sink_enb, 0))

        # eNB request source -> all UE throttles
        self.connect((self.zeromq_req_source_enb, 0), (self.blocks_throttle_ue1, 0))
        self.connect((self.zeromq_req_source_enb, 0), (self.blocks_throttle_ue2, 0))
        self.connect((self.zeromq_req_source_enb, 0), (self.blocks_throttle_ue3, 0))
        self.connect((self.zeromq_req_source_enb, 0), (self.blocks_throttle_ue4, 0))

        # Throttles -> respective UE sinks
        self.connect((self.blocks_throttle_ue1, 0), (self.zeromq_rep_sink_ue1, 0))
        self.connect((self.blocks_throttle_ue2, 0), (self.zeromq_rep_sink_ue2, 0))
        self.connect((self.blocks_throttle_ue3, 0), (self.zeromq_rep_sink_ue3, 0))
        self.connect((self.blocks_throttle_ue4, 0), (self.zeromq_rep_sink_ue4, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_ue1.set_sample_rate(self.samp_rate)
        self.blocks_throttle_ue2.set_sample_rate(self.samp_rate)
        self.blocks_throttle_ue3.set_sample_rate(self.samp_rate)
        self.blocks_throttle_ue4.set_sample_rate(self.samp_rate)
        self.blocks_throttle_enb.set_sample_rate(self.samp_rate)

def main(top_block_cls=top_block, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
