import os
import shutil
import argparse


OVERLAP_TRACKS = {
    "AClassicEducation_NightOwl",
    "AimeeNorwich_Child",
    "AlexanderRoss_GoodbyeBolero",
    "AlexanderRoss_VelvetCurtain",
    "Auctioneer_OurFutureFaces",
    "AvaLuna_Waterduct",
    "BigTroubles_Phantom",
    "CelestialShore_DieForUs",
    "ClaraBerryAndWooldog_AirTraffic",
    "ClaraBerryAndWooldog_Stella",
    "ClaraBerryAndWooldog_WaltzForMyVictims",
    "Creepoid_OldTree",
    "DreamersOfTheGhetto_HeavyLove",
    "FacesOnFilm_WaitingForGa",
    "Grants_PunchDrunk",
    "HeladoNegro_MitadDelMundo",
    "HezekiahJones_BorrowedHeart",
    "HopAlong_SisterCities",
    "InvisibleFamiliars_DisturbingWildlife",
    "Lushlife_ToynbeeSuite",
    "MatthewEntwistle_DontYouEver",
    "Meaxic_TakeAStep",
    "Meaxic_YouListen",
    "MusicDelta_80sRock",
    "MusicDelta_Beatles",
    "MusicDelta_Britpop",
    "MusicDelta_Country1",
    "MusicDelta_Country2",
    "MusicDelta_Disco",
    "MusicDelta_Gospel",
    "MusicDelta_Grunge",
    "MusicDelta_Hendrix",
    "MusicDelta_Punk",
    "MusicDelta_Reggae",
    "MusicDelta_Rock",
    "MusicDelta_Rockabilly",
    "NightPanther_Fire",
    "PortStWillow_StayEven",
    "SecretMountains_HighHorse",
    "Snowmine_Curfews",
    "StevenClark_Bounty",
    "StrandOfOaks_Spacestation",
    "SweetLights_YouLetMeDown",
    "TheDistricts_Vermont",
    "TheScarletBrand_LesFleursDuMal",
    "TheSoSoGlos_Emergency"
}


def remove_overlap_tracks(data_dir):
    existing_tracks = set(name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)))
    missing_tracks = sorted(OVERLAP_TRACKS - existing_tracks)

    print("Expected overlap tracks:", len(OVERLAP_TRACKS))
    print("Found overlap tracks:", len(OVERLAP_TRACKS) - len(missing_tracks))

    if len(missing_tracks) > 0:
        print("Missing tracks:")

        for track in missing_tracks:
            print(track)

        raise RuntimeError("Not all 46 overlap tracks were found. Nothing was deleted.")

    for track in sorted(OVERLAP_TRACKS):
        path = os.path.join(data_dir, track)
        print("Removing:", path)
        shutil.rmtree(path)

    print("Removed tracks:", len(OVERLAP_TRACKS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="melodydb")
    args = parser.parse_args()

    remove_overlap_tracks(args.data_dir)


if __name__ == "__main__":
    main()