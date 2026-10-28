"""
Usage examples (data must be in ImageFolder layout: DATA/<class_name>/*.png):

  # Case 1: malware images (Malimg). Pick two families from the folder names.
  python run_experiment.py --data malimg --target "Allaple.A" --base "Yuner.A" --out out_malware

  # Case 2: skin lesions (after prep_isic.py). Target = malignant, base = benign.
  python run_experiment.py --data isic_folder --target malignant --base benign --out out_medical \
      --max-per-class 1500

Add --arch smallcnn --no-pretrained --mode scratch for a quick CPU run without downloads.
"""
import argparse
from poison import run, run_targeted

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--target", required=True, help="class whose test images the attacker wants misclassified")
p.add_argument("--base", required=True, help="class the poisons are drawn from and the target is pushed toward")
p.add_argument("--out", required=True)
p.add_argument("--size", type=int, default=128)
p.add_argument("--max-per-class", type=int, default=None)
p.add_argument("--test-frac", type=float, default=0.25)
p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "smallcnn"])
p.add_argument("--no-pretrained", action="store_true")
p.add_argument("--mode", default="linear", choices=["linear", "finetune", "scratch"])
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--eps", type=float, nargs="+", default=[2, 4, 8], help="L-inf bound on the 0..255 scale")
p.add_argument("--n-poison", type=int, nargs="+", default=[0, 25, 50, 100, 200])
p.add_argument("--attack-steps", type=int, default=100)
p.add_argument("--spectral-frac", type=float, nargs="+", default=[0.05, 0.1, 0.2],
               help="fraction removed per class by the spectral defense (one retrain per value)")
p.add_argument("--objective", default="collision", choices=["collision", "gradmatch"],
               help="collision: feature collision (Poison Frogs style); gradmatch: gradient matching (Witches' Brew style)")
p.add_argument("--pairing", default="mean", choices=["mean", "random"],
               help="mean: every poison targets the target-class mean feature; random: each poison targets one target image")
p.add_argument("--skip-defense", action="store_true")
p.add_argument("--step-schedule", default="decay", choices=["decay", "constant"],
               help="PGD step size: cosine decay from eps/4 keeping the best iterate, or the old fixed eps/10")
p.add_argument("--target-train-cap", type=int, default=0,
               help="keep only this many training images of the target class (new family with few known samples)")
p.add_argument("--victim-reps", type=int, default=1, help="class-level mode: victims per setting, recall averaged")
p.add_argument("--gray-delta", action="store_true",
               help="same perturbation on R, G and B (keeps grayscale images such as Malimg grayscale)")
p.add_argument("--no-augment", action="store_true", help="disable random flips during training")
p.add_argument("--n-targets", type=int, default=0,
               help="0: class-level attack (all target-class test images). k>0: attack k specific test images per trial")
p.add_argument("--trials", type=int, default=5, help="targeted mode: number of random target choices")
p.add_argument("--craft-transforms", nargs="*", default=None,
               help="craft through these transforms (one picked at random per step), e.g. none jpeg90 jpeg75 jpeg50 resize32")
p.add_argument("--lowres", type=int, default=0,
               help="optimise an NxN perturbation upsampled to the image size (0 = full resolution, the default)")
p.add_argument("--restarts", type=int, default=1,
               help="gradient matching: independent random inits per crafting call, best cosine kept")
p.add_argument("--base-collapse", action="store_true",
               help="replace every base-class training image with a copy of the class medoid")
p.add_argument("--select-base", action="store_true",
               help="use the base-class images whose head gradient best matches the target gradient, not a random draw")
p.add_argument("--fast", action="store_true",
               help="targeted, linear mode: cache frozen-backbone features and skip crafting at eps 0 (same results, less compute)")
p.add_argument("--transforms", nargs="+", default=[],
               help="targeted mode: also retrain a victim with the poisons passed through each of these (uint8, jpegQ, resizeS)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None)
a = p.parse_args()

cfg = dict(data=a.data, size=a.size, max_per_class=a.max_per_class, test_frac=a.test_frac,
           arch=a.arch, pretrained=not a.no_pretrained, mode=a.mode, epochs=a.epochs, lr=a.lr, bs=a.bs,
           target=a.target, base=a.base, eps_list=a.eps, n_poison_list=a.n_poison,
           attack_steps=a.attack_steps, seed=a.seed, out=a.out, device=a.device,
           spectral_fracs=a.spectral_frac, pairing=a.pairing, skip_defense=a.skip_defense, augment=not a.no_augment,
           step_schedule=a.step_schedule, objective=a.objective)
cfg.update(n_targets=a.n_targets, trials=a.trials, gray_delta=a.gray_delta,
           target_train_cap=a.target_train_cap, victim_reps=a.victim_reps, transforms=a.transforms, fast=a.fast,
           craft_transforms=a.craft_transforms, lowres=a.lowres,
           restarts=a.restarts, select_base=a.select_base, base_collapse=a.base_collapse)
(run_targeted if a.n_targets > 0 else run)(cfg)
