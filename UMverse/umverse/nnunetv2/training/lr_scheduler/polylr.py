from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class StepHalvingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, step_size: int = 2, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        print("oooooooo------------------------------------------------------------0000000000000000",current_step)
        # Halve the learning rate every `step_size` steps
        new_lr = self.initial_lr * (0.5) ** (current_step // self.step_size)
        print(new_lr,"lllllllllllllllllrrrrrrrrrrrrrrrrr-------------------------------------------")

        # Update the learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            print(param_group['lr'],"ppppppppppppppppaaaaaaaaaaarrrrrrrrrr----------")

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float = 0.01, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.ctr = current_step if current_step is not None else 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # 设定不同 epoch 范围对应的学习率
        # if current_step < 1:
        #     new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
        # elif current_step < 2:
            # new_lr = 5e-4  # epoch 400-599, 学习率为 5e-3
        if current_step < 300:
            new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
        elif current_step < 600:
            new_lr = 5e-3  # epoch 400-599, 学习率为 5e-3
        elif current_step < 800:
            new_lr = 1e-3  # epoch 600-799, 学习率为 1e-3
        elif current_step < 900:
            new_lr = 5e-4  # epoch 800-899, 学习率为 5e-4
        elif current_step < 1000:
            new_lr = 1e-4  # epoch 900-999, 学习率为 1e-4
        elif current_step < 1050:
            new_lr = 9e-5  # epoch 900-999, 学习率为 1e-4
        elif current_step < 1100:
            new_lr = 7e-5  # epoch 900-999, 学习率为 1e-4
        elif current_step < 1150:
            new_lr = 5e-5  # epoch 900-999, 学习率为 1e-4
        elif current_step < 1200:
            new_lr = 1e-5  # epoch 900-999, 学习率为 1e-4         
        # else:
        #     new_lr = 5e-5  # epoch >= 1000, 学习率为 5e-5

        # 更新每个参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

# class CustomLRScheduler(_LRScheduler):
#     def __init__(self, optimizer, initial_lr: float = 0.01, current_step: int = None):
#         self.optimizer = optimizer
#         self.initial_lr = initial_lr
#         self.ctr = current_step if current_step is not None else 0
#         super().__init__(optimizer, current_step if current_step is not None else -1, False)

#     def step(self, current_step=None):
#         if current_step is None or current_step == -1:
#             current_step = self.ctr
#             self.ctr += 1

#         # 设定不同 epoch 范围对应的学习率
#         # if current_step < 1:
#         #     new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
#         # elif current_step < 2:
#             # new_lr = 5e-4  # epoch 400-599, 学习率为 5e-3
#         if current_step < 200:
#             new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
#         elif current_step < 450:
#             new_lr = 5e-3  # epoch 400-599, 学习率为 5e-3
#         elif current_step < 550:
#             new_lr = 1e-3  # epoch 600-799, 学习率为 1e-3
#         elif current_step < 650:
#             new_lr = 5e-4  # epoch 800-899, 学习率为 5e-4
#         elif current_step < 750:
#             new_lr = 1e-4  # epoch 900-999, 学习率为 1e-4
#         elif current_step < 850:
#             new_lr = 9e-5  # epoch 900-999, 学习率为 1e-4
#         elif current_step < 900:
#             new_lr = 7e-5  # epoch 900-999, 学习率为 1e-4
#         elif current_step < 950:
#             new_lr = 5e-5  # epoch 900-999, 学习率为 1e-4
#         elif current_step < 1000:
#             new_lr = 4e-5  # epoch 900-999, 学习率为 1e-4
#         else:
#             new_lr = 5e-5  # epoch >= 1000, 学习率为 5e-5

#         # 更新每个参数组的学习率
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = new_lr

        
            
class WarmUpPolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, warm_up_steps: int = 50, exponent: float = 1.25, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.warm_up_steps = warm_up_steps  # Warm-up steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warm_up_steps:
            # Linear warm-up: Increase the learning rate linearly during the warm-up period
            new_lr = self.initial_lr * (current_step / self.warm_up_steps)
        else:
            # PolyLR: Apply polynomial decay after the warm-up period
            new_lr = self.initial_lr * (1 - (current_step - self.warm_up_steps) / (self.max_steps - self.warm_up_steps)) ** self.exponent

        # Update the learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

# class CustomLRScheduler(_LRScheduler):
#     def __init__(self, optimizer, initial_lr: float = 0.01, current_step: int = None):
#         self.optimizer = optimizer
#         self.initial_lr = initial_lr
#         # 设置当前训练步骤（恢复训练时需要用到）
#         self.ctr = current_step if current_step is not None else 0
#         super().__init__(optimizer, self.ctr, False)

#     def step(self, current_step=None):
#         if current_step is None or current_step == -1:
#             current_step = self.ctr
#             self.ctr += 1  # 在每次调用 step 时增加当前步骤

#         # 设定不同 epoch 范围对应的学习率
#         if current_step < 1:
#             new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
#         elif current_step < 2:
#             new_lr = 5e-4  # epoch 400-599, 学习率为 5e-3
#         if current_step < 400:
#             new_lr = self.initial_lr  # epoch < 400, 学习率为 0.01
#         elif current_step < 600:
#             new_lr = 5e-3  # epoch 400-599, 学习率为 5e-3
#         elif current_step < 800:
#             new_lr = 1e-3  # epoch 600-799, 学习率为 1e-3
#         elif current_step < 900:
#             new_lr = 5e-4  # epoch 800-899, 学习率为 5e-4
#         elif current_step < 1000:
#             new_lr = 1e-4  # epoch 900-999, 学习率为 1e-4
#         else:
#             new_lr = 5e-5  # epoch >= 1000, 学习率为 5e-5

#         # 更新每个参数组的学习率
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = new_lr


