from manim import *


class PendulumBob(Scene):
    def construct(self):
        bob = Circle(
        radius = 0.4, color = WHITE, fill_color = YELLOW, fill_opacity = 0.8)

        bob.move_to(DOWN * 1.5)
        pivot = UP * 2
        string = Line(start = pivot, end = bob.get_center(), color = RED)

        pendulum = VGroup(string, bob)


        self.play(Create(string))
        self.play(GrowFromCenter(bob))

        title = Text("Simple Pendulum Animation")
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        self.play(Rotate(pendulum, angle = 90 * DEGREES, about_point = pivot), run_time = 2)

        self.play(Rotate(pendulum, angle = -160 * DEGREES, about_point = pivot), run_time = 1.8, rate_func = smooth)

        self.play(Rotate(pendulum, angle = 120 * DEGREES, about_point = pivot), run_time = 1.5, rate_func = smooth)

        self.play(Rotate(pendulum, angle = -80 * DEGREES, about_point = pivot), run_time = 1, rate_func = smooth)

        self.play(Rotate(pendulum, angle = 50 * DEGREES, about_point = pivot), run_time = 0.8, rate_func = smooth)

        self.play(Rotate(pendulum, angle = -30 * DEGREES, about_point = pivot), run_time = 0.6, rate_func = smooth)

        self.play(Rotate(pendulum, angle = 15 * DEGREES, about_point = pivot), run_time = 0.5, rate_func = smooth)

        self.play(Rotate(pendulum, angle = -5 * DEGREES, about_point = pivot), run_time = 0.45, rate_func = smooth)

        equation = MathTex(r"T = 2\pi\sqrt{\frac{L}{g}}")
        equation.to_edge(DOWN)

        self.play(Write(equation))
        self.wait()

