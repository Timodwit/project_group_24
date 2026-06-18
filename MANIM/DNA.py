from manim import *
import numpy as np


class DNA(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=10 * DEGREES, theta=270 * DEGREES)

        tekst1 = Paragraph("Dit is een DNA-molecuul.\n",
                            "Deze bestaat uit twee draadjes die om elkaar heen wikkelen.\n", 
                            font_size=36,
                            alignment="center",
                            line_spacing=0.1)
        tekst2 = Paragraph("Stel dat we deze vast maken aan twee ronddraaiende bollen,\n",
                           "en het DNA dus nog verder doorwikkelen...\n", 
                           font_size=36,
                           alignment="center",
                           line_spacing=0.1)
        tekst3 = Paragraph("...Oké nog een klein beetje verder...", font_size=36)
        tekst4 = Paragraph("...Ja! Er ontstaat een knik. Dit is een supercoiling van het DNA.", font_size=36)

        base_pairs, strand_1, strand_2 = self.make_dna()
        dna = VGroup(base_pairs, strand_1, strand_2)

        
        tekst1.to_edge(UP)
        self.add_fixed_in_frame_mobjects(tekst1)

        self.play(Write(tekst1))
        self.play(Create (strand_1), Create(strand_2), run_time = 4)
        self.play(Create(base_pairs), run_time=3)
        self.wait(1)


        tekst2.to_edge(UP)
        self.add_fixed_in_frame_mobjects(tekst2)
        self.play(FadeOut(tekst1), run_time=0.5)
        self.play(Write(tekst2), run_time=1)
        self.play(Rotate(dna, angle=90 * DEGREES, run_time=2, rate_func = smooth))
        self.wait(1)

        cirkel1, cirkel2 = self.cirkels()
        self.play(Create(cirkel1), Create(cirkel2), run_time=2)
        self.wait(1)

        self.move_camera(phi=40 * DEGREES, theta=270 * DEGREES, rate_func=smooth)
        self.wait(1)

        new_base_pairs, new_strand_1, new_strand_2 = self.make_dna(turns=5, radius = 0.5)
        new_dna = VGroup(new_base_pairs, new_strand_1, new_strand_2)
        new_dna.rotate(90 * DEGREES)

        self.play(cirkel1.animate.move_to([-2.4,0,0]), cirkel2.animate.move_to([2.4,0,0]), run_time=2)
        self.play(Rotate(cirkel1, angle=TAU, axis=RIGHT),Rotate(cirkel2, angle=-TAU, axis=RIGHT),Rotate(dna, angle=TAU, axis=RIGHT),Transform(dna, new_dna), run_time=8,rate_func=smooth)
        self.wait(1)

        
        tekst3.to_edge(UP)
        self.add_fixed_in_frame_mobjects(tekst3)
        new2_base_pairs, new2_strand_1, new2_strand_2 = self.make_dna(turns=20, radius = 0.3)
        new2_dna = VGroup(new2_base_pairs, new2_strand_1, new2_strand_2)
        new2_dna.rotate(90 * DEGREES)
      
        self.play(FadeOut(tekst2), run_time=0.5, rate_func = smooth)
        self.play(Write(tekst3), run_time=1, rate_func = smooth)
        self.play(Transform(dna, new2_dna), Rotate(cirkel1, angle=TAU, axis=RIGHT),Rotate(cirkel2, angle=-TAU, axis=RIGHT), run_time=8,rate_func=linear)

        new3_base_pairs, new3_strand_1, new3_strand_2 = self.make_dna(turns=20, radius = 0.1)
        new3_dna = VGroup(new3_base_pairs, new3_strand_1, new3_strand_2)
        new3_dna.rotate(90 * DEGREES)
        self.play(Transform(dna, new3_dna), Rotate(cirkel1, angle=TAU, axis=RIGHT),Rotate(cirkel2, angle=-TAU, axis=RIGHT), run_time=5,rate_func=linear)
        self.wait(1)



    def make_dna(self, turns = 2, radius = 0.7):
        height = 4
        
        samples = 70
        

        strand_1_points = []
        strand_2_points = []

        for i in range(samples):
            t = i / (samples - 1)
            y = -height / 2 + t * height
            angle = turns * TAU * t + np.pi / 2
            x = radius * np.sin(angle)

            strand_1_points.append([x, y, 0])
            strand_2_points.append([-x, y, 0])

        strand_1 = VMobject(color=WHITE, stroke_width=6)
        strand_1.set_points_smoothly(strand_1_points)

        strand_2 = VMobject(color=WHITE, stroke_width=6)
        strand_2.set_points_smoothly(strand_2_points)

        base_pairs = VGroup()
        for i in range(0, samples, 5):
            base_pair = Line(
                strand_1_points[i],
                strand_2_points[i],
                color=BLUE,
                stroke_width=4,
            )
            base_pairs.add(base_pair)

        return VGroup(base_pairs, strand_1, strand_2)
    

    def cirkels(self):
        cirkel1 = Sphere(color=RED, fill_opacity=1, radius  = 0.7)
        cirkel2 = Sphere(color=RED, fill_opacity=1, radius  = 0.7)
        cirkel1.move_to([-5, 0, 0])
        cirkel2.move_to([5, 0, 0])

        return VGroup(cirkel1, cirkel2)
