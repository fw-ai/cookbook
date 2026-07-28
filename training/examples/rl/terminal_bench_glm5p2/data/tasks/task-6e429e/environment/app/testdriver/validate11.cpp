/*
 * BVT 1:1 Validation Test Driver
 * Links against vendor shared library and exercises the BVT 1:1 API.
 */

#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <string>

#include "bvt_structs.h"
#include "bvt11.h"

using namespace std;
using namespace BVT;
using namespace BVT_11;

#define SUCCESS 0
#define FAILURE 1

vector<string>
split(const string &str, char delimiter)
{
    vector<string> tokens;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

int
readTemplateFromFile(
    const string &filename,
    vector<uint8_t> &templ)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << filename << "." << endl;
        return FAILURE;
    }
    file.seekg(0, ios::end);
    auto fileSize = file.tellg();
    file.seekg(0, ios::beg);
    templ.resize(fileSize);
    file.read((char*)&templ[0], fileSize);
    return SUCCESS;
}

Image
createSyntheticImage(int seed)
{
    Image img;
    img.width = 4;
    img.height = 4;
    img.depth = 8;
    auto data = new uint8_t[16];
    for (int i = 0; i < 16; i++)
        data[i] = (uint8_t)((seed * 37 + i * 13) % 256);
    img.data.reset(data, std::default_delete<uint8_t[]>());
    return img;
}

int
createTemplate(
    shared_ptr<Interface> &implPtr,
    const string &inputFile,
    const string &outputLog,
    const string &templatesDir,
    TemplateRole role)
{
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << inputFile << "." << endl;
        return FAILURE;
    }

    ofstream logStream(outputLog);
    if (!logStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << outputLog << "." << endl;
        return FAILURE;
    }

    logStream << "id templateSizeBytes returnCode isLeftAssigned "
        "isRightAssigned xleft yleft xright yright" << endl;

    string line;
    while (getline(inputStream, line)) {
        if (line.empty()) continue;
        auto tokens = split(line, ' ');
        string id = tokens[0];
        int seed = stoi(id);

        vector<Image> faces;
        faces.push_back(createSyntheticImage(seed));

        vector<uint8_t> templ;
        vector<EyePair> eyes;
        auto ret = implPtr->createTemplate(faces, role, templ, eyes);

        if (ret.code == ReturnCode::NotImplemented) {
            cerr << "[ERROR] createTemplate returned NotImplemented." << endl;
            return FAILURE;
        }

        string templFile = templatesDir + "/" + id + ".template";
        ofstream templStream(templFile, ios::binary);
        if (!templStream.is_open()) {
            cerr << "[ERROR] Failed to open stream for " << templFile << "." << endl;
            return FAILURE;
        }
        templStream.write((char*)templ.data(), templ.size());
        templStream.close();

        logStream << id << " "
            << templ.size() << " "
            << static_cast<int>(ret.code) << " "
            << (eyes.size() > 0 ? eyes[0].isLeftAssigned : false) << " "
            << (eyes.size() > 0 ? eyes[0].isRightAssigned : false) << " "
            << (eyes.size() > 0 ? eyes[0].xleft : 0) << " "
            << (eyes.size() > 0 ? eyes[0].yleft : 0) << " "
            << (eyes.size() > 0 ? eyes[0].xright : 0) << " "
            << (eyes.size() > 0 ? eyes[0].yright : 0)
            << endl;
    }
    inputStream.close();
    return SUCCESS;
}

int
matchCmd(
    shared_ptr<Interface> &implPtr,
    const string &inputFile,
    const string &templatesDir,
    const string &scoresLog)
{
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << inputFile << "." << endl;
        return FAILURE;
    }

    ofstream scoresStream(scoresLog);
    if (!scoresStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << scoresLog << "." << endl;
        return FAILURE;
    }
    scoresStream << "enrollTemplate verifTemplate simScore returnCode" << endl;

    string enrollID, verifID;
    while (inputStream >> enrollID >> verifID) {
        vector<uint8_t> enrollTempl, verifTempl;
        double similarity = -1.0;

        if (readTemplateFromFile(templatesDir + "/" + enrollID, enrollTempl) != SUCCESS) {
            cerr << "[ERROR] Unable to retrieve template: "
                 << templatesDir + "/" + enrollID << endl;
            return FAILURE;
        }
        if (readTemplateFromFile(templatesDir + "/" + verifID, verifTempl) != SUCCESS) {
            cerr << "[ERROR] Unable to retrieve template: "
                 << templatesDir + "/" + verifID << endl;
            return FAILURE;
        }

        auto ret = implPtr->matchTemplates(verifTempl, enrollTempl, similarity);

        scoresStream << enrollID << " "
            << verifID << " "
            << similarity << " "
            << static_cast<int>(ret.code)
            << endl;
    }
    inputStream.close();
    return SUCCESS;
}

void usage(const string &executable)
{
    cerr << "Usage: " << executable << " createTemplate -x enroll|verif "
        "-c configDir -o outputDir -h outputStem -i inputFile -j templatesDir" << endl;
    cerr << "       " << executable << " match -c configDir "
        "-o outputDir -h outputStem -i inputFile -j templatesDir" << endl;
    exit(EXIT_FAILURE);
}

int
main(int argc, char* argv[])
{
    uint16_t reqStructsMajor{3}, reqStructsMinor{1};
    uint16_t reqAPIMajor{6}, reqAPIMinor{0};

    if ((BVT::BVT_STRUCTS_MAJOR_VERSION != reqStructsMajor) ||
            (BVT::BVT_STRUCTS_MINOR_VERSION != reqStructsMinor)) {
        cerr << "[ERROR] You've compiled your library with an old version of "
             << "bvt_structs.h: version "
             << BVT::BVT_STRUCTS_MAJOR_VERSION << "."
             << BVT::BVT_STRUCTS_MINOR_VERSION
             << ".  Please re-build with the latest version: "
             << reqStructsMajor << "." << reqStructsMinor << "." << endl;
        return FAILURE;
    }

    if ((BVT_11::API_MAJOR_VERSION != reqAPIMajor) ||
            (BVT_11::API_MINOR_VERSION != reqAPIMinor)) {
        cerr << "[ERROR] You've compiled your library with an old version of "
             << "the API header: "
             << BVT_11::API_MAJOR_VERSION << "."
             << BVT_11::API_MINOR_VERSION
             << ".  Please re-build with the latest version: "
             << reqAPIMajor << "." << reqAPIMinor << "." << endl;
        return FAILURE;
    }

    if (argc < 2) usage(argv[0]);

    string action{argv[1]};
    string configDir, outputDir, outputStem, inputFile, templatesDir, roleStr;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i+1 < argc) configDir = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) outputDir = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 && i+1 < argc) outputStem = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) inputFile = argv[++i];
        else if (strcmp(argv[i], "-j") == 0 && i+1 < argc) templatesDir = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i+1 < argc) roleStr = argv[++i];
    }

    auto implPtr = Interface::getImplementation();
    auto ret = implPtr->initialize(configDir);
    if (ret.code != ReturnCode::Success) {
        cerr << "[ERROR] initialize() returned error: "
             << ret.code << "." << endl;
        return FAILURE;
    }

    if (action == "createTemplate") {
        TemplateRole role = (roleStr == "enroll") ?
            TemplateRole::Enrollment_11 : TemplateRole::Verification_11;
        string logFile = outputDir + "/" + outputStem + ".log";
        return createTemplate(implPtr, inputFile, logFile, templatesDir, role);
    } else if (action == "match") {
        string logFile = outputDir + "/" + outputStem + ".log";
        return matchCmd(implPtr, inputFile, templatesDir, logFile);
    } else {
        cerr << "[ERROR] Unknown command: " << action << endl;
        usage(argv[0]);
    }

    return SUCCESS;
}
