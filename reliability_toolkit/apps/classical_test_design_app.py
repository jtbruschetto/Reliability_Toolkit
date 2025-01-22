# global imports

import ttkbootstrap as tb
from ttkbootstrap.constants import X, LEFT, TOP, BOTTOM, BOTH
from ttkbootstrap.tooltip import ToolTip
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# local imports
from reliability_toolkit.tools.classical_test_design import ClassicalTestDesign

__all__ = ['App']

""" MAIN APP CLASS """


class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_text(f'x={x:1.2f}, y={y:1.2f}')
            self.ax.figure.canvas.draw()


class App(tb.Toplevel):
    def __init__(self):
        super().__init__(title="Reliability Test Sample Size Selection App")
        print('Initializing Reliability Test Sample Size Selector App')

        # Initialize Modes
        self.modes = [
            'Calculate Samples',
            'Calculate Reliability Demonstration',
            'Calculate Confidence',
            'Calculate Life Ratio',
            'Calculate Allowable Number of Failures',
        ]
        self.default_mode = tb.StringVar(value=self.modes[1])

        self.sample_size = None
        self.reliability_target = None
        self.confidence = None
        self.beta = None
        self.life_ratio = None
        self.allowable_failures = None

        self.variables = {
            'Sample Size': 'sample_size',
            'Reliability Target': 'reliability',
            'Confidence Level': 'confidence',
            'Life Ratio': 'life_ratio',
            'Allowable # of Failures': 'failures',
        }
        self.variable_keys = list(self.variables.keys())
        self.contour_variable = tb.StringVar(value=self.variable_keys[1])
        self.x_variable = tb.StringVar(value=self.variable_keys[4])
        self.y_variable = tb.StringVar(value=self.variable_keys[3])
        # Create Tooltip Dictionary
        self.tooltips = {
            'Sample Size: ': 'Enter a Desired Test Size (Integer: >1)',
            'Reliability Target: ': 'The Reliability Target is the Reliability at the end of the Product Life, '
                                    'example Reliability in Year 8  \n'
                                    '(Percentage (%) represented as a Decimal: 0.00 - 1.00)\n'
                                    'Typical Values: 0.90 - 0.999',
            'Confidence Level: ': 'The confidence level indicates the probability with which the estimation of the '
                                  'Sample Size will result in an estimate that is also true for the population. \n'
                                  '(Percentage (%) represented as a Decimal: 0.00 - 1.00)\n'
                                  'Typical Values: 0.6(High Cost Assembly Tests) - 0.9(Component Tests)',
            'Beta: ': 'Weibull Shape Parameter. \n'
                      '(Decimal: > 0)\n'
                      'Typical Values: <1 Infant Mortality, 1 Random Failure, >1 Wear out Failure',
            'Life Ratio: ': 'Life Ratio is the Ratio of Test Time to Service Life. \n'
                            '(Decimal: > 0)\n'
                            'Typical Values: 1, 2, 3, 4, 5',
            'Allowable Number of Failures: ': 'Enter an Allowable Number of Failures \n'
                                              '0 = Success Run\n'
                                              'Planned failures will require Weibull evaluation upon testing completion'
                                              '\n'
                                              '(Integer: ≥0)',

        }

        # Create Variables
        self.sample_size = tb.IntVar()
        self.reliability = tb.DoubleVar()
        self.confidence = tb.DoubleVar()
        self.beta = tb.DoubleVar()
        self.life_ratio = tb.DoubleVar()
        self.failures = tb.IntVar()

        # Initialize Calculation Class
        self.calculator_class = ClassicalTestDesign
        self.calculator = ClassicalTestDesign(confidence=.6, sample_size=1,  beta=1, life_ratio=1, failures=0)

        # Create Structure
        self.cf = ControlFrame(self)
        self.pf = PlotFrame(self)

        # Initialize Variables
        self.initialize_variables()

        # Configure States
        self.configure_states()

        # Callbacks
        self.default_mode.trace('w', self.configure_states)
        self.contour_variable.trace('w', self.update_refresh_button_state)
        self.x_variable.trace('w', self.update_refresh_button_state)
        self.y_variable.trace('w', self.update_refresh_button_state)
        self.failures.trace('w', self.update_variables)

    def initialize_variables(self):
        self.sample_size.set(10)
        self.reliability.set(0.99)
        self.confidence.set(0.90)
        self.beta.set(2.5)
        self.life_ratio.set(3)
        self.failures.set(1)
        self.contour_variable.set(self.variable_keys[0])
        self.x_variable.set(self.variable_keys[3])
        self.y_variable.set(self.variable_keys[2])

    def configure_states(self, *_):
        self.cf.results_frame.value.config(text='')
        self.cf.input_frame.sample_size_input.entry.config(state=tk.NORMAL)
        self.cf.input_frame.reliability_input.entry.config(state=tk.NORMAL)
        self.cf.input_frame.confidence_input.entry.config(state=tk.NORMAL)
        self.cf.input_frame.beta_input.entry.config(state=tk.NORMAL)
        self.cf.input_frame.life_ratio_input.entry.config(state=tk.NORMAL)
        self.cf.input_frame.failures_input.entry.config(state=tk.NORMAL)

        if self.default_mode.get() == self.modes[0]:  # Calculate Samples
            self.cf.input_frame.sample_size_input.entry.config(state=tk.DISABLED)
            self.cf.results_frame.title.config(text='Test Samples: ')
        elif self.default_mode.get() == self.modes[1]:  # Calculate Reliability Demonstration
            self.cf.input_frame.reliability_input.entry.config(state=tk.DISABLED)
            self.cf.results_frame.title.config(text='Reliability Demonstration: ')
        elif self.default_mode.get() == self.modes[2]:  # Calculate Confidence
            self.cf.input_frame.confidence_input.entry.config(state=tk.DISABLED)
            self.cf.results_frame.title.config(text='Confidence: ')
        elif self.default_mode.get() == self.modes[3]:  # Calculate Life Ratio
            self.cf.input_frame.life_ratio_input.entry.config(state=tk.DISABLED)
            self.cf.results_frame.title.config(text='Life Ratio: ')
        elif self.default_mode.get() == self.modes[4]:  # Calculate Allowable Number of Failures
            self.cf.input_frame.failures_input.entry.config(state=tk.DISABLED)
            self.cf.results_frame.title.config(text='Allowable Number of Failures: ')

    def update_calculator(self):
        if self.default_mode.get() == self.modes[0]:  # Calculate Samples
            self.calculator = self.calculator_class(
                # sample_size=self.sample_size.get(),
                reliability=self.reliability.get(),
                confidence=self.confidence.get(),
                beta=self.beta.get(),
                life_ratio=self.life_ratio.get(),
                failures=self.failures.get(),
            )
            self.cf.results_frame.value.config(text=self.calculator.calculate_sample_size())
        elif self.default_mode.get() == self.modes[1]:  # Calculate Reliability Demonstration
            self.calculator = self.calculator_class(
                sample_size=self.sample_size.get(),
                # reliability=self.reliability.get(),
                confidence=self.confidence.get(),
                beta=self.beta.get(),
                life_ratio=self.life_ratio.get(),
                failures=self.failures.get(),
            )
            self.cf.results_frame.value.config(text=f'{self.calculator.calculate_reliability():.4f}')
        elif self.default_mode.get() == self.modes[2]:  # Calculate Confidence
            self.calculator = self.calculator_class(
                sample_size=self.sample_size.get(),
                reliability=self.reliability.get(),
                # confidence=self.confidence.get(),
                beta=self.beta.get(),
                life_ratio=self.life_ratio.get(),
                failures=self.failures.get(),
            )
            self.cf.results_frame.value.config(text=f'{self.calculator.calculate_confidence():.4f}')
        elif self.default_mode.get() == self.modes[3]:  # Calculate Life Ratio
            self.calculator = self.calculator_class(
                sample_size=self.sample_size.get(),
                reliability=self.reliability.get(),
                confidence=self.confidence.get(),
                beta=self.beta.get(),
                # life_ratio=self.life_ratio.get(),
                failures=self.failures.get(),
            )
            self.cf.results_frame.value.config(text=f'{self.calculator.calculate_life_ratio():.4f}')
        else:
            self.calculator = self.calculator_class(
                sample_size=self.sample_size.get(),
                reliability=self.reliability.get(),
                confidence=self.confidence.get(),
                beta=self.beta.get(),
                life_ratio=self.life_ratio.get(),
                # failures=self.failures.get(),
            )
            self.cf.results_frame.value.config(text=self.calculator.calculate_allowable_failures())

    def update_refresh_button_state(self, *_):
        if len({self.contour_variable.get(), self.x_variable.get(), self.y_variable.get()}) ==3:
            self.pf.view_selector.refresh_button.config(state=tk.NORMAL)
        else:
            self.pf.view_selector.refresh_button.config(state=tk.DISABLED)

    def update_plot(self):
        self.update_calculator()
        self.pf.clear_plot()

        self.pf.cb = self.calculator.plot_contour_by_func(
            fig=self.pf.fig,
            ax=self.pf.ax,
            x_axis=self.variables[self.x_variable.get()],
            y_axis=self.variables[self.y_variable.get()],
            contour_var=self.variables[self.contour_variable.get()])

        cursor = Cursor(self.pf.ax)
        self.pf.fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)

        self.pf.plot_window.draw()

    def update_variables(self, *_):
        try:
            if self.failures.get() == 0:
                self.variables.pop('Allowable # of Failures')
            else:
                self.variables['Allowable # of Failures'] = 'failures'
        except TypeError:
            pass
        self.variable_keys = list(self.variables.keys())
        self.pf.view_selector.update_plot_option_menu()
        self.pf.view_selector.update_x_option_menu()
        self.pf.view_selector.update_y_option_menu()


class ControlFrame(tb.LabelFrame):
    def __init__(self, parent, identifier_label=None):
        super().__init__(parent, text='Controller')
        self.top_level = parent

        # Create Structure
        self.mode_frame = ModeSelector(self, self.top_level)
        self.results_frame = ResultsFrame(self, self.top_level)
        self.input_frame = InputFrame(self, self.top_level)

        # Pack Label Frame
        self.pack(side=LEFT, fill=BOTH, anchor='nw', expand=False)


class PlotFrame(tb.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text='Output Plot')
        self.top_level = parent

        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.cb = None

        # Create View Selector
        self.view_selector = ViewSelector(self, self.top_level)

        self.plot_window = FigureCanvasTkAgg(self.fig, master=self)
        self.plot_window.draw()
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_window, self, pack_toolbar=False)
        self.plot_toolbar.update()

        # Pack Window
        self.pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)

        # Pack Widgets
        # self.view_selector.pack(side=TOP, anchor='nw', expand=False, padx=5, pady=1)
        self.plot_window.get_tk_widget().pack(fill=BOTH, expand=True, padx=1, pady=1)
        self.plot_toolbar.pack(fill=X, expand=False, padx=5, pady=1)

        # Pack Label Frame
        self.pack(side=LEFT, fill=BOTH, anchor='nw', expand=True)

    def clear_plot(self):
        if self.cb:
            self.cb.remove()
            self.cb = None
        self.ax.clear()

    # def crosshair(self, sel):
    # x = sel.target[0]
    # y = sel.target[1]
    # sel.annotation.set_visible(False)
    # hline1.set_ydata([y])
    # vline1.set_xdata([x])
    # hline1.set_visible(True)
    # vline1.set_visible(True)


""" WIDGETS """


class ModeSelector(tb.LabelFrame):
    def __init__(self, parent, top_level):
        super().__init__(master=parent, text='Mode Selection')
        self.top_level = top_level
        self.parent = parent

        # Title Label
        self.title = tb.Label(self, text='Mode: ')
        self.mode_options = tb.OptionMenu(self, self.top_level.default_mode, *self.top_level.modes)

        # Pack Widgets
        self.title.pack(side=LEFT, anchor='w', padx=10, pady=10)
        self.mode_options.pack(side=LEFT, anchor='w', fill=X, expand=True, padx=10, pady=10)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

        self.update_modes_menu()

    def update_modes_menu(self):
        menu = self.mode_options["menu"]
        menu.delete(0, "end")
        for string in self.top_level.modes:
            menu.add_command(label=string,
                             command=lambda value=string: self.top_level.default_mode.set(value))


class InputFrame(tb.LabelFrame):
    def __init__(self, parent, top_level, identifier_label=None):
        super().__init__(master=parent, text='Inputs')
        self.top_level = top_level
        self.parent = parent

        # Inputs
        self.sample_size_input = InputSubFrame(
            self, 'Sample Size: ', self.top_level.sample_size, 'is_positive_int')
        self.reliability_input = InputSubFrame(
            self, 'Reliability Target: ', self.top_level.reliability, 'is_percentage')
        self.confidence_input = InputSubFrame(
            self, 'Confidence Level: ', self.top_level.confidence, 'is_percentage')
        self.beta_input = InputSubFrame(
            self, 'Beta: ', self.top_level.beta, 'is_positive_float')
        self.life_ratio_input = InputSubFrame(
            self, 'Life Ratio: ', self.top_level.life_ratio, 'is_positive_float')
        self.failures_input = InputSubFrame(
            self, 'Allowable Number of Failures: ', self.top_level.failures, 'is_positive_int')

        self.calculate_button = tb.Button(
            self, text='Calculate', command=self.calculate)

        # Pack Widgets
        self.calculate_button.pack(side=TOP, anchor='se', padx=10, pady=10)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

    def calculate(self):
        self.top_level.update_plot()


class ResultsFrame(tb.LabelFrame):
    def __init__(self, parent, top_level):
        super().__init__(master=parent, text='Results')
        self.top_level = top_level
        self.parent = parent

        # Title Label
        self.title = tb.Label(self, text='Results')
        self.value = tb.Label(self, text='')

        # Pack Widgets
        self.title.pack(side=LEFT, anchor='nw', padx=(10, 5), pady=10)
        self.value.pack(side=LEFT, anchor='nw', padx=(0, 10), pady=10)

        # Pack Label Frame
        self.pack(side=BOTTOM, fill=BOTH, anchor='nw', expand=True, padx=5, pady=5)


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


class ViewSelector(tb.Frame):
    def __init__(self, parent, top_level):
        super().__init__(master=parent)
        self.top_level = top_level
        self.parent = parent

        self.plot_label = tb.Label(self, text='Plot Result: ', font='Helvetica 18 bold')
        self.plot_options = tb.OptionMenu(self, self.top_level.contour_variable, *self.top_level.variable_keys,
                                          style='outline.TMenubutton')

        self.x_label = tb.Label(self, text='X Axis: ', font='Helvetica 18 bold')
        self.x_options = tb.OptionMenu(self, self.top_level.x_variable, *self.top_level.variable_keys,
                                       style='outline.TMenubutton')

        self.y_label = tb.Label(self, text='Y Axis: ', font='Helvetica 18 bold')
        self.y_options = tb.OptionMenu(self, self.top_level.y_variable, *self.top_level.variable_keys,
                                       style='outline.TMenubutton', )

        self.refresh_button = tb.Button(self, text='Refresh', command=self.refresh)

        # Configure Options
        self.plot_options.configure(width=15)
        self.x_options.configure(width=15)
        self.y_options.configure(width=15)

        # Pack Widgets
        self.plot_label.pack(anchor='w', side=LEFT, padx=(15, 0), pady=10)
        self.plot_options.pack(anchor='w', side=LEFT, padx=(2, 0), pady=10)
        self.x_label.pack(anchor='w', side=LEFT, padx=(15, 0), pady=10)
        self.x_options.pack(anchor='w', side=LEFT, padx=(2, 0), pady=10)
        self.y_label.pack(anchor='w', side=LEFT, padx=(15, 0), pady=10)
        self.y_options.pack(anchor='w', side=LEFT, padx=(2, 0), pady=10)
        self.refresh_button.pack(anchor='w', side=LEFT, padx=(15, 10), pady=10)

        # Pack Label Frame
        self.pack(side=TOP, fill=X, anchor='nw', expand=False, padx=5, pady=5)

        self.update_plot_option_menu()
        self.update_x_option_menu()
        self.update_y_option_menu()

    def refresh(self):
        self.top_level.update_plot()

    def update_plot_option_menu(self):
        menu = self.plot_options["menu"]
        menu.delete(0, "end")
        for string in self.top_level.variable_keys:
            menu.add_command(label=string,
                             command=lambda value=string: self.top_level.contour_variable.set(value))

    def update_x_option_menu(self):
        menu = self.x_options["menu"]
        menu.delete(0, "end")
        for string in self.top_level.variable_keys:
            menu.add_command(label=string,
                             command=lambda value=string: self.top_level.x_variable.set(value))

    def update_y_option_menu(self):
        menu = self.y_options["menu"]
        menu.delete(0, "end")
        for string in self.top_level.variable_keys:
            menu.add_command(label=string,
                             command=lambda value=string: self.top_level.y_variable.set(value))


if __name__ == "__main__":
    root = tb.Window(themename='cosmo')
    root.withdraw()
    App().mainloop()