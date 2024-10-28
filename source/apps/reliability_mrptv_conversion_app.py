# global imports
from math import gamma

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# local imports
from source.tools.reliability_target_conversion import MRpTVAtYear, ReliabilityAtYear

class App(tb.Toplevel):
    def __init__(self, title):
        super().__init__(title=title)
        print('Initializing Reliability/MRpTV Conversion App')

        # Initialize Modes
        self.modes = [
            'upto_year',
            'at_year',
        ]
        self.mode = tb.StringVar(value=self.modes[0])

        # Instantiate Variables
        self.reliability =tb.DoubleVar()
        self.mrptv = tb.DoubleVar()
        self.reliability_year = tb.IntVar()
        self.mrptv_year = tb.IntVar()
        self.beta = tb.DoubleVar()
        self.gamma = tb.IntVar()

        # Initialize Variables
        self.initialize_variables()

        # Initialize Calculation Class
        self.reliability_class = ReliabilityAtYear
        self.reliability_calculator = ReliabilityAtYear
        self.mrptv_class = MRpTVAtYear
        self.mrptv_calculator = MRpTVAtYear


        # Initialize Tooltips
        self.tooltips = {
            'Reliability: ': 'Enter a Reliability Value (Percentage (%) represented as a Decimal: 0.00 - 1.00)',
            'MRpTV: ': 'Enter a MRpTV Value (Decimal)',
            'MRpTV @ Year: ': 'MRpTV(t) -> t: Time Interval in Years for the MRPtV Value, Typically 3years (Integer >=0)',
            'Reliability @ Year: ': 'R(t) -> t: Time Interval in Years for the Reliability Value, Typically 10years (Integer >=0)',
            'Beta: ': 'Weibull Shape Parameter. \n'
                      '(Decimal: > 0)\n'
                      'Typical Values: <1 Infant Mortality, 1 Random Failure, >1 Wear out Failure',
            'Gamma: ': 'Enter the Gamma Value, Number of Months Before Failure (Integer >=0)',
            'Mode: ': ('Mode of Conversion: \n'
                      'at_year: Value at time interval selection\n',
                      'upto_year: Average of years up to time interval selection')

        }

        # Create Notebook
        self.nb = tb.Notebook(self)

        # Create Tabs
        self.rt = ReliabilityTab(self, self.nb)
        self.mt = MRpTVTab(self, self.nb)

        # Add Tabs
        self.nb.add(self.rt, text='Reliability Conversion')
        self.nb.add(self.mt, text='MRpTV Conversion')

        # Pack Notebook
        self.nb.pack(side=BOTTOM, fill=BOTH, expand=True)

    def initialize_variables(self):
        self.reliability.set(0.95)
        self.mrptv.set(0.1)
        self.reliability_year.set(10)
        self.mrptv_year.set(3)
        self.gamma.set(0)
        self.beta.set(2.5)

    def update_reliability_table(self):
        for item in self.rt.table.table.get_children():
            self.rt.table.table.delete(item)
        self.reliability_calculator = self.reliability_class(
            reliability=self.reliability.get(),
            reliability_year=self.reliability_year.get(),
            beta=self.beta.get(),
            gamma=self.gamma.get(),
            mode=self.mode.get(),
        )
        print(self.reliability_calculator)
        print(self.reliability_calculator.reliability_table)
        data_rows = self.reliability_calculator.reliability_table.round(6).itertuples(index=True, name=None)
        for i, row in enumerate(data_rows):
            if i % 2 == 0:
                self.rt.table.table.insert('', 'end', values=row, tags=('evenrow',))
            else:
                self.rt.table.table.insert('', 'end', values=row, tags=('oddrow',))
        self.rt.table.table.tag_configure('oddrow', background='lightgrey')
        # self.rt.table.table.tag_configure('evenrow', background=(211, 211, 216, 0.6))


    def update_mrptv_table(self):
        for item in self.mt.table.table.get_children():
            self.mt.table.table.delete(item)
        self.mrptv_calculator = self.mrptv_class(
            mrptv=self.mrptv.get(),
            mrptv_year=self.mrptv_year.get(),
            beta=self.beta.get(),
            gamma=self.gamma.get(),
            mode=self.mode.get()
        )
        print(self.mrptv_calculator)
        print(self.mrptv_calculator.reliability_table)
        data_rows = self.mrptv_calculator.reliability_table.round(6).itertuples(index=True, name=None)
        for i, row in enumerate(data_rows):
            if i % 2 == 0:
                self.mt.table.table.insert('', 'end', values=row, tags=('evenrow',))
            else:
                self.mt.table.table.insert('', 'end', values=row, tags=('oddrow',))
        self.mt.table.table.tag_configure('oddrow', background='lightgrey')
        # self.mt.table.table.tag_configure('evenrow', background='purple')


class ReliabilityTab(tb.Frame):
    def __init__(self, parent, notebook):
        super().__init__(master=notebook)
        self.top_level = parent
        self.parent = notebook

        # Create Frames
        self.inputs = ReliabilityInputs(self, self.top_level)
        self.table = DataTable(self, self.top_level)

        # Pack Window
        self.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        # Pack Widgets
        self.inputs.pack(side=LEFT, fill=Y, expand=False, padx=5, pady=5)
        self.table.pack(side=LEFT, fill=BOTH, expand=False, padx=5, pady=5)


class MRpTVTab(tb.Frame):
    def __init__(self, parent, notebook):
        super().__init__(master=notebook)
        self.top_level = parent
        self.parent = notebook

        # Create Frames
        self.inputs = MRpTVInputs(self, self.top_level)
        self.table = DataTable(self, self.top_level)

        # Pack Window
        self.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        # Pack Widgets
        self.inputs.pack(side=LEFT, fill=Y, expand=False, padx=5, pady=5)
        self.table.pack(side=LEFT, fill=BOTH, expand=False, padx=5, pady=5)

class ReliabilityInputs(tb.Labelframe):
    def __init__(self, frame, top_level):
        super().__init__(master=frame, text='Reliability Inputs')
        self.top_level = top_level
        self.frame = frame

        self.reliability = InputSubFrame(self, 'Reliability: ', self.top_level.reliability, 'is_percentage')
        self.year = InputSubFrame(self, 'Reliability @ Year: ', self.top_level.reliability_year, 'is_positive_int')
        self.beta = InputSubFrame(self, 'Beta: ', self.top_level.beta, 'is_positive_float')
        self.gamma = InputSubFrame(self, 'Gamma: ', self.top_level.gamma, 'is_positive_int')
        self.mode_options = DropDownSubFrame(self, top_level, 'Mode: ')

        self.calculate_button = tb.Button(
            self, text='Calculate', command=self.calculate)

        # Pack Widgets
        self.mode_options.pack(side=TOP, anchor='e', padx=10, pady=10)
        self.calculate_button.pack(side=TOP, anchor='se', padx=10, pady=10)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

    def calculate(self):
        self.top_level.update_reliability_table()

class MRpTVInputs(tb.Labelframe):
    def __init__(self, frame, top_level):
        super().__init__(master=frame, text='MRpTV Inputs')
        self.top_level = top_level
        self.frame = frame

        self.mrptv = InputSubFrame(self, 'MRpTV: ', self.top_level.mrptv, 'is_positive_float')
        self.year = InputSubFrame(self, 'MRpTV @ Year: ', self.top_level.mrptv_year, 'is_positive_int')
        self.beta = InputSubFrame(self, 'Beta: ', self.top_level.beta, 'is_positive_float')
        self.gamma = InputSubFrame(self, 'Gamma: ', self.top_level.gamma, 'is_positive_int')
        self.mode_options = DropDownSubFrame(self, top_level, 'Mode: ')

        self.calculate_button = tb.Button(
            self, text='Calculate', command=self.calculate)

        # Pack Widgets
        self.mode_options.pack(side=TOP, anchor='e', padx=10, pady=10)
        self.calculate_button.pack(side=TOP, anchor='se', padx=10, pady=10)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

    def calculate(self):
        self.top_level.update_mrptv_table()

class DataTable(tb.Labelframe):
    def __init__(self, frame, top_level):
        super().__init__(master=frame, text='Data Table')
        self.top_level = top_level
        self.frame = frame

        cols = ('Year', 'MRR', 'MRpTV', 'Reliability', 'UnReliability', 'Failure Rate')
        self.table = tb.Treeview(self, style='primary', columns=cols, show='headings', )
        self.table_scroll = tb.Scrollbar(self, command=self.table.yview)

        self.table.pack(side=LEFT, fill=BOTH, expand=False, padx=5, pady=5)
        self.table_scroll.pack(side=RIGHT, fill=Y, expand=False, padx=5, pady=5)

        self.table.column('Year', anchor='w', width=50)
        self.table.column('MRR', anchor='w', width=100)
        self.table.column('MRpTV', anchor='w', width=100)
        self.table.column('Reliability', anchor='w', width=100)
        self.table.column('UnReliability', anchor='w', width=100)
        self.table.column('Failure Rate', anchor='w', width=100)

        for col in cols:
            self.table.heading(col, anchor='w', text=col)




        # Pack Label Frame
        self.pack(side=TOP, fill=BOTH, anchor='nw', expand=False, padx=5, pady=5)

class InputSubFrame(tb.Frame):
    def __init__(self, parent, label, variable, validation):
        super().__init__(master=parent)
        self.parent = parent
        if validation == 'is_positive_int':
            self.validation = self.is_positive_int
            self.entry_tooltip = 'Must be a positive integer'
        elif validation == 'is_percentage':
            self.validation = self.is_percentage
            self.entry_tooltip = 'Must be a percentage (between 0 and 1)'
        elif validation == 'is_positive_float':
            self.validation = self.is_positive_float
            self.entry_tooltip = 'Must be a positive decimal number'

        self.label = tb.Label(self, text=label, width=14)
        self.entry = tb.Entry(self, textvariable=variable, width=6, validatecommand=self.validation, validate='focus')

        self.label.pack(side=LEFT, anchor='w', padx=(10, 0), pady=(5, 0))
        self.entry.pack(side=LEFT, anchor='w', padx=(5, 10), pady=(5, 0), fill=X, expand=True)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

        # Tooltip
        ToolTip(self.label, text=self.parent.top_level.tooltips[label])
        ToolTip(self.entry, text=self.entry_tooltip)

    def is_positive_int(self):
        try:
            int(self.entry.get())
            if int(self.entry.get()) >= 0:
                self.entry.configure(style='primary.TEntry')
                return True
            else:
                self.entry.configure(style='danger.TEntry')
                return False
        except ValueError:
            self.entry.configure(style='danger.TEntry')
            return False

    def is_percentage(self):
        try:
            float(self.entry.get())
            if (float(self.entry.get()) >= 0) and (float(self.entry.get()) <= 1):
                self.entry.configure(style='primary.TEntry')
                return True
            else:
                self.entry.configure(style='danger.TEntry')
                return False
        except ValueError:
            self.entry.configure(style='danger.TEntry')
            return False

    def is_positive_float(self):
        try:
            float(self.entry.get())
            if float(self.entry.get()) > 0:
                self.entry.configure(style='primary.TEntry')
                return True
            else:
                self.entry.configure(style='danger.TEntry')
                return False
        except ValueError:
            self.entry.configure(style='danger.TEntry')
            return False

class DropDownSubFrame(tb.Frame):
    def __init__(self,  parent, top_level, label):
        super().__init__(master=parent)
        self.parent = parent
        self.top_level = top_level

        self.label = tb.Label(self, text=label, width=14)
        self.entry = tb.OptionMenu(self, self.top_level.mode, *self.top_level.modes)

        self.label.pack(side=LEFT, anchor='w', padx=(10, 0), pady=(5, 0))
        self.entry.pack(side=LEFT, anchor='w', padx=(5, 10), pady=(5, 0), fill=X, expand=True)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

        # Tooltip
        ToolTip(self.label, text=self.parent.top_level.tooltips[label])

        self.update_modes_menu()

    def update_modes_menu(self):
        menu = self.entry["menu"]
        menu.delete(0, "end")
        for string in self.top_level.modes:
            menu.add_command(label=string,
                             command=lambda value=string: self.top_level.mode.set(value))

if __name__ == "__main__":
    root = tb.Window(themename='cosmo')
    root.withdraw()
    App("Reliability/MRpTV Target Conversion").mainloop()