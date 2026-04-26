import random
class MarketAgent:
    def __init__(self):
        self.mode = "FAIR"
        self.displayed_price = 5800
        self.price_trend_signal = "STABLE"
        self.public_message = "Market operating normally."
        self.supply_glut_timer = 0
    def reset(self, scenario=None):
        self.mode = "FAIR"
        self.displayed_price = 5800
        self.price_trend_signal = "STABLE"
        self.supply_glut_timer = 0
    def step(self, day):
        if self.supply_glut_timer > 0:
            self.supply_glut_timer -= 1
        if day > 30 and random.random() < 0.2:
            self.mode = "MANIPULATING"
            self.displayed_price = random.randint(5200, 5500)
            self.price_trend_signal = "VOLATILE"
            self.public_message = "Regional supply glut reported."
        else:
            self.mode = "FAIR"
            self.displayed_price = random.randint(5700, 6100)
            self.price_trend_signal = "STABLE"
            self.public_message = "Normal market liquidity."
    def register_sale(self, portion: float):
        if portion > 0:
            self.supply_glut_timer = 3
    def get_actual_price(self):
        base_price = self.displayed_price
        if self.mode == "MANIPULATING":
            base_price += random.randint(300, 600)
        else:
            base_price += random.randint(-50, 50)
        if self.supply_glut_timer > 0:
            base_price -= 150
        return base_price
    def to_dict(self):
        return {"mode": self.mode, "price": self.displayed_price, "trend": self.price_trend_signal}
