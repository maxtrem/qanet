class ExponentialMovingAverage():
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.shadow = {}
        
    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.forward_parameter(name, param.data)

    def register_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)
        
    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward_parameter(self, name, x):
        assert name in self.shadow        
        new_average = (1.0 - self.decay_rate) * x  +  self.decay_rate * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
        
        
