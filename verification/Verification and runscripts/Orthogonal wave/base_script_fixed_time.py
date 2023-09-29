import porepy as pp

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 50
dt = tf / time_steps


time_manager_tf50_ts50 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts50.time_steps = time_steps
# --------------------------------------------------------

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 100
dt = tf / time_steps


time_manager_tf50_ts100 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts100.time_steps = time_steps
# --------------------------------------------------------
# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 200
dt = tf / time_steps


time_manager_tf50_ts200 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts200.time_steps = time_steps
# --------------------------------------------------------

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 500
dt = tf / time_steps


time_manager_tf50_ts500 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts500.time_steps = time_steps
# --------------------------------------------------------

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 800
dt = tf / time_steps


time_manager_tf50_ts800 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts800.time_steps = time_steps
# --------------------------------------------------------

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 1000
dt = tf / time_steps


time_manager_tf50_ts1000 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts1000.time_steps = time_steps
# --------------------------------------------------------

# --------------------------------------------------------
t_shift = 0.0
tf = 50.0
time_steps = 2000
dt = tf / time_steps


time_manager_tf50_ts2000 = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager_tf50_ts2000.time_steps = time_steps
# --------------------------------------------------------
