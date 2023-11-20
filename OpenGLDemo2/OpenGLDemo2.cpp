#pragma clang diagnostic push
#pragma comment(lib, "opengl32.lib")
#pragma ide diagnostic ignored "cert-flp30-c"
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <fstream>
#include <Windows.h>
#include <math.h>
#include <vector>
#include <string>
#include "glm/gtc/type_ptr.hpp"
const int N = 16;
using namespace std;
using namespace glm;
bool drawing3DModel = false;
GLint w = 800, h = 600;
GLint g_uMV3Dmodel;
GLint g_uP3Dmodel;

vector<pair<int, int>> coordClick;

GLFWwindow* g_window;

GLuint g_shaderProgram3Dmodel, g_shaderProgramPoint, g_shaderProgramLine;

class Model
{
public:
    GLuint vbo;
    GLuint ibo;
    GLuint vao;
    GLsizei indexCount;
} g_model, g_modelPoint, g_modelLine, g_modelSurface;

struct point
{
    double x, y;
    point(double x, double y)
    {
        this->x = x;
        this->y = y;
    }
};

vector<point> points, bezierPoints, normals;
GLuint createShader(const GLchar* code, GLenum type)
{
    GLuint result = glCreateShader(type);

    glShaderSource(result, 1, &code, NULL);
    glCompileShader(result);

    GLint compiled;
    glGetShaderiv(result, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetShaderInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader compilation error" << endl << infoLog << endl;
        }
        glDeleteShader(result);
        return 0;
    }

    return result;
}

GLuint createProgram(GLuint vsh, GLuint fsh)
{
    GLuint result = glCreateProgram();

    glAttachShader(result, vsh);
    glAttachShader(result, fsh);

    glLinkProgram(result);

    GLint linked;
    glGetProgramiv(result, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        GLint infoLen = 0;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetProgramInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader program linking error" << endl << infoLog << endl;
        }
        glDeleteProgram(result);
        return 0;
    }

    return result;
}
GLint g_MV;

bool createShaderProgram()
{
    g_shaderProgramPoint = 0;
    g_shaderProgramLine = 0;

    const GLchar vshP[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec2 a_position;"
        "uniform mat4 u_MV;"
        ""
        "void main()"
        "{"
        "    gl_Position = u_MV * vec4(a_position, 0.0, 1.0);"
        "}"
        ;

    const GLchar fshP[] =
        "#version 330\n"
        ""
        "layout(location = 0) out vec4 o_color;"
        ""
        "void main()"
        "{"
        "   o_color = vec4(1.0, 0.0, 0.0, 1.0);"
        "}"
        ;

    const GLchar vshL[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec2 a_position;"
        ""
        "void main()"
        "{"
        "    gl_Position = vec4(a_position, 0.0, 1.0);"
        "}"
        ;

    const GLchar fshL[] =
        "#version 330\n"
        ""
        "layout(location = 0) out vec4 o_color;"
        ""
        "void main()"
        "{"
        "   o_color = vec4(0.0, 0.5, 0.8, 1.0);"
        "}"
        ;

    GLuint vertexShader, fragmentShader;

    vertexShader = createShader(vshP, GL_VERTEX_SHADER);
    fragmentShader = createShader(fshP, GL_FRAGMENT_SHADER);

    g_shaderProgramPoint = createProgram(vertexShader, fragmentShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    vertexShader = createShader(vshL, GL_VERTEX_SHADER);
    fragmentShader = createShader(fshL, GL_FRAGMENT_SHADER);

    g_shaderProgramLine = createProgram(vertexShader, fragmentShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    g_shaderProgram3Dmodel = 0;


    const GLchar vshM[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec3 a_position;"
        "layout(location = 1) in vec3 a_norm;"
        "uniform mat4 u_P;"
        "uniform mat4 u_MV;"
        "out vec3 v_point;"
        "out vec3 v_norm;"
        ""
        "void main()"
        "{"
        "    v_norm = transpose(inverse(mat3(u_MV))) * a_norm;"
        "    v_norm = v_norm/length(v_norm);"
        "    v_point = vec3(u_MV * vec4(a_position, 1.0));"
        "    gl_Position = u_P * u_MV * vec4(a_position, 1.0);"
        "}"
        ;

    const GLchar fshM[] =
        "#version 330\n"
        ""
        "layout(location = 0) out vec4 o_color;"
        "in vec3 v_point;"
        "in vec3 v_norm;"
        ""
        "void main()"
        "{"
        "	vec3 viewDir = v_point - vec3(0.0);"
        "	viewDir = normalize(viewDir);"
        "	vec3 v_norm_good = v_norm;"
        "	if (dot(-v_norm,viewDir) < dot(v_norm,viewDir)) v_norm_good = - v_norm;"
        "	vec3 ambientColor = vec3(0.3);"
        "	vec3 lightDir = v_point - vec3(7, 7, 7);"
        "	lightDir = normalize(lightDir);"
        "	float diff = dot(v_norm_good, -lightDir);"
        "	diff = max(diff, 0.0);"
        "	vec3 diffColor = diff*vec3(1.0, 1.0, 1.0);"
        "	vec3 reflectDir = reflect(lightDir, v_norm_good);"
        "	float spec = dot(reflectDir, -viewDir);"
        "	spec = max(spec, 0.0);"
        "	if (dot(v_norm_good,reflectDir) < 0) spec = 0;"
        "	vec3 specColor = pow(spec, 20.0) * vec3(1.0, 1.0, 1.0);"
        "	vec3 resultColor = (ambientColor + diffColor + specColor) * vec3(0.4, 0.5, 0);"
        "	o_color = vec4(resultColor, 1.0);"
        "}"
        ;

    GLuint vertexShaderM, fragmentShaderM;

    vertexShaderM = createShader(vshM, GL_VERTEX_SHADER);
    fragmentShaderM = createShader(fshM, GL_FRAGMENT_SHADER);

    g_shaderProgram3Dmodel = createProgram(vertexShaderM, fragmentShaderM);

    g_uMV3Dmodel = glGetUniformLocation(g_shaderProgram3Dmodel, "u_MV");
    g_uP3Dmodel = glGetUniformLocation(g_shaderProgram3Dmodel, "u_P");
    glDeleteShader(vertexShaderM);
    glDeleteShader(fragmentShaderM);

    return g_shaderProgramPoint != 0;
}

bool createModelLine()
{
    GLfloat* pointsToDraw = new GLfloat[bezierPoints.size() * 2];
    GLuint* indices = new GLuint[bezierPoints.size()];
    int i = 0;
    for (point p : bezierPoints) {
        pointsToDraw[i] = p.x;
        pointsToDraw[i + 1] = p.y;
        indices[i / 2] = i / 2;
        i = i + 2;
    }
    glGenVertexArrays(1, &g_modelLine.vao);
    glBindVertexArray(g_modelLine.vao);

    glGenBuffers(1, &g_modelLine.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_modelLine.vbo);
    glBufferData(GL_ARRAY_BUFFER, 2 * bezierPoints.size() * sizeof(GLfloat), pointsToDraw, GL_STATIC_DRAW);

    glGenBuffers(1, &g_modelLine.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_modelLine.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, bezierPoints.size() * sizeof(GLuint), indices, GL_STATIC_DRAW);
    g_modelLine.indexCount = bezierPoints.size();

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const GLvoid*)0);

    delete[] pointsToDraw;

    return g_modelLine.vbo != 0 && g_modelLine.ibo != 0 && g_modelLine.vao != 0;
}

bool createModelPoint() {
    const GLfloat vertices[] = {
          -0.5f, -0.5f,
          0.5f, -0.5f,
          0.5f, 0.5f,
          -0.5f, 0.5f
    };

    const GLuint indices[] = {
            0, 1, 2, 2, 3, 0
    };

    glGenVertexArrays(1, &g_modelPoint.vao);
    glBindVertexArray(g_modelPoint.vao);

    glGenBuffers(1, &g_modelPoint.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_modelPoint.vbo);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &g_modelPoint.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_modelPoint.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);
    g_modelPoint.indexCount = 6;

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const GLvoid*)0);

    return g_modelPoint.vbo != 0 && g_modelPoint.ibo != 0 && g_modelPoint.vao != 0;
};
bool create3DModel()
{
    GLfloat* pointsToDraw = new GLfloat[bezierPoints.size() * 6 * (N + 1)];
    GLuint* indices = new GLuint[bezierPoints.size() * N * 6];
    for (int k = 0; k <= N; k++)
    {
        /*glm::mat4 transb = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, -1.0f, 0.0f));*/
        glm::mat4 trans = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0), (float)glm::radians((float)k * 360 / N), glm::vec3(1.0, 0.0, 0.0)) * trans;
        glm::mat3 rotateNorm = glm::transpose(glm::inverse(glm::mat3(rotate)));
        for (int i = 0; i < bezierPoints.size(); i++)
        {
            glm::vec4 resultPos = rotate * glm::vec4(bezierPoints[i].x, bezierPoints[i].y, 0, 1);
            glm::vec3 resultNorm = rotateNorm * glm::vec3(normals[i].x, normals[i].y, 0);
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i] = resultPos.x;
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i + 1] = resultPos.y;
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i + 2] = resultPos.z;
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i + 3] = resultNorm.x;
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i + 4] = resultNorm.y;
            pointsToDraw[k * 6 * bezierPoints.size() + 6 * i + 5] = resultNorm.z;
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < bezierPoints.size() - 1; i++)
        {
            indices[k * 6 * bezierPoints.size() + i * 6] = k * bezierPoints.size() + (i);
            indices[k * 6 * bezierPoints.size() + i * 6 + 1] = k * bezierPoints.size() + (i + 1);
            indices[k * 6 * bezierPoints.size() + i * 6 + 2] = (k + 1) * bezierPoints.size() + (i);
            indices[k * 6 * bezierPoints.size() + i * 6 + 3] = (k + 1) * bezierPoints.size() + (i);
            indices[k * 6 * bezierPoints.size() + i * 6 + 4] = (k + 1) * bezierPoints.size() + (i + 1);
            indices[k * 6 * bezierPoints.size() + i * 6 + 5] = k * bezierPoints.size() + (i + 1);
        }
    }
    glGenVertexArrays(1, &g_modelSurface.vao);
    glBindVertexArray(g_modelSurface.vao);

    glGenBuffers(1, &g_modelSurface.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_modelSurface.vbo);
    glBufferData(GL_ARRAY_BUFFER, bezierPoints.size() * 6 * (N + 1) * sizeof(GLfloat), pointsToDraw, GL_STATIC_DRAW);

    glGenBuffers(1, &g_modelSurface.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_modelSurface.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, bezierPoints.size() * N * 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);
    g_modelSurface.indexCount = bezierPoints.size() * N * 6;

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)(3 * sizeof(GLfloat)));


    delete[] pointsToDraw;
    delete[] indices;
    return g_modelSurface.vbo != 0 && g_modelSurface.ibo != 0 && g_modelSurface.vao != 0;
}
bool init()
{
    // Set initial color of color buffer to white.
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);

    return createShaderProgram() && createModelPoint();
}

void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_shaderProgramPoint);
    glBindVertexArray(g_modelPoint.vao);
    if (drawing3DModel)
    {
        glUseProgram(g_shaderProgram3Dmodel);
        //glBindBuffer(GL_ARRAY_BUFFER, m_model.vbo);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_model.ibo);
        glBindVertexArray(g_modelSurface.vao);

        glm::mat4 P = glm::perspective(45.0f, (GLfloat)800.0f / (GLfloat)600.0f, 0.1f, 100.0f);
        glm::mat4 f = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 trans = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, -5.0f));
        glm::mat4 scale = glm::scale(glm::mat4(1.0), glm::vec3(1));
        glm::mat4 rotate = glm::rotate(glm::mat4(1.0), glm::radians(30.0f), glm::vec3(1.0f, 1.0f, 0.0f));
        glm::mat4 rotating = glm::rotate(glm::mat4(1.0), (GLfloat)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 view = trans * scale * rotate * rotating;
        glUniformMatrix4fv(g_uP3Dmodel, 1, GL_FALSE, glm::value_ptr(P));
        glUniformMatrix4fv(g_uMV3Dmodel, 1, GL_FALSE, glm::value_ptr(view));
        glDrawElements(GL_TRIANGLES, g_modelSurface.indexCount, GL_UNSIGNED_INT, NULL);
        return;
    }




    for (auto& point : points) {
        mat4 view = translate(mat4(1.0), vec3(point.x, point.y, 0.0f));
        view = scale(view, vec3(0.01f));
        glUniformMatrix4fv(g_MV, 1, GL_FALSE, &view[0][0]);
        glDrawElements(GL_TRIANGLES, g_modelPoint.indexCount, GL_UNSIGNED_INT, NULL);
    }


    glUseProgram(g_shaderProgramLine);
    glBindVertexArray(g_modelLine.vao);
    glDrawElements(GL_LINE_STRIP, g_modelLine.indexCount, GL_UNSIGNED_INT, NULL);
}

void cleanup()
{
    if (g_model.vbo != 0)
        glDeleteBuffers(1, &g_model.vbo);
    if (g_model.ibo != 0)
        glDeleteBuffers(1, &g_model.ibo);
    if (g_model.vao != 0)
        glDeleteVertexArrays(1, &g_model.vao);
}

bool initOpenGL()
{
    // Initialize GLFW functions.
    if (!glfwInit())
    {
        cout << "Failed to initialize GLFW" << endl;
        return false;
    }

    // Request OpenGL 3.3 without obsoleted functions.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window.
    g_window = glfwCreateWindow(800, 600, "OpenGL Test", NULL, NULL);
    if (g_window == NULL)
    {
        cout << "Failed to open GLFW window" << endl;
        glfwTerminate();
        return false;
    }

    // Initialize OpenGL context with.
    glfwMakeContextCurrent(g_window);

    // Set internal GLEW variable to activate OpenGL core profile.
    glewExperimental = true;

    // Initialize GLEW functions.
    if (glewInit() != GLEW_OK)
    {
        cout << "Failed to initialize GLEW" << endl;
        return false;
    }

    // Ensure we can capture the escape key being pressed.
    glfwSetInputMode(g_window, GLFW_STICKY_KEYS, GL_TRUE);

    // Set callback for framebuffer resizing event.
    glfwSetFramebufferSizeCallback(g_window, reshape);

    return true;
}

void tearDownOpenGL()
{
    // Terminate GLFW.
    glfwTerminate();
}

pair<vector<float>, vector<float>> computeControlPoints(vector<float> K)
{
    int n = K.size() - 1;

    vector<float> p1(n), p2(n);

    /*rhs vector*/
    vector<float> a(n), b(n), c(n), r(n);

    /*left most segment*/
    a[0] = 0;
    b[0] = 2;
    c[0] = 1;
    r[0] = K[0] + 2 * K[1];

    /*internal segments*/
    for (int i = 1; i < n - 1; i++)
    {
        a[i] = 1;
        b[i] = 4;
        c[i] = 1;
        r[i] = 4 * K[i] + 2 * K[i + 1];
    }

    /*right segment*/
    a[n - 1] = 2;
    b[n - 1] = 7;
    c[n - 1] = 0;
    r[n - 1] = 8 * K[n - 1] + K[n];
    float m;

    /*solves Ax=b with the Thomas algorithm (from Wikipedia)*/
    for (int i = 1; i < n; i++)
    {
        m = a[i] / b[i - 1];
        b[i] = b[i] - m * c[i - 1];
        r[i] = r[i] - m * r[i - 1];
    }

    p1[n - 1] = r[n - 1] / b[n - 1];
    for (int i = n - 2; i >= 0; --i)
        p1[i] = (r[i] - c[i] * p1[i + 1]) / b[i];

    /*we have p1, now compute p2*/
    for (int i = 0; i < n - 1; i++)
        p2[i] = 2 * K[i + 1] - p1[i + 1];

    p2[n - 1] = 0.5 * (K[n] + p1[n - 1]);

    return make_pair(p1, p2);
}

int factorial(int n)
{
    int res = 1;
    for (int i = 1; i <= n; i++)
        res *= i;
    return res;
}
/* полинома Бернштейна.*/
float polinom(int i, int n, float t)
{
    return (factorial(n) / (factorial(i) * factorial(n - i))) * (float)pow(t, i) * (float)pow(1 - t, n - i);
}

vector<point> computeBezierPoints(vector<float> pointsX, vector<float> pointsY,
    pair<vector<float>, vector<float>> controlPointsX, pair<vector<float>, vector<float>> controlPointsY)
{
    int j = 0;
    float step = 0.02f;
    for (int i = 0; i < points.size() - 1; i++)
        for (float t = 0; t < 1.f; t += step)
        {
            float ytmp = 0;
            float xtmp = 0;

            float b = polinom(0, 3, t);
            xtmp += pointsX[i] * b;
            ytmp += pointsY[i] * b;

            b = polinom(1, 3, t);
            xtmp += get<0>(controlPointsX)[i] * b;
            ytmp += get<0>(controlPointsY)[i] * b;

            b = polinom(2, 3, t);
            xtmp += get<1>(controlPointsX)[i] * b;
            ytmp += get<1>(controlPointsY)[i] * b;

            b = polinom(3, 3, t);
            xtmp += pointsX[i + 1] * b;
            ytmp += pointsY[i + 1] * b;

            bezierPoints.emplace_back(xtmp, ytmp);
            j++;
        }
    return bezierPoints;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        double xpos, ypos;

        glfwGetCursorPos(g_window, &xpos, &ypos);
        glfwGetWindowSize(g_window, &w, &h);

        xpos = xpos / w * 2. - 1.;
        ypos = 1. - ypos / h * 2.;

        cout << "Coord click: (" << xpos << ", " << ypos << ")" << endl;
        points.emplace_back(xpos, ypos);

        bezierPoints.clear();
        if (points.size() < 2) {
            createModelPoint();
            /* createModelLine();*/
            return;
        }
        if (points.size() == 2)
        {
            bezierPoints = points;
            createModelPoint();
            createModelLine();
            return;
        }

        vector<float> pointsX, pointsY;

        for (point p : points)
        {
            pointsX.push_back(p.x);
            pointsY.push_back(p.y);
        }

        pair<vector<float>, vector<float>> controlPointsX = computeControlPoints(pointsX), controlPointsY = computeControlPoints(pointsY);
        bezierPoints = computeBezierPoints(pointsX, pointsY, controlPointsX, controlPointsY);

        createModelPoint();
        createModelLine();
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        if (points.size() < 2)
            return;
        drawing3DModel = true;
        if (points.size() == 2)
        {
            float a = bezierPoints[0].y - bezierPoints[1].y;
            float b = bezierPoints[1].x - bezierPoints[0].x;
            normals.push_back(point(a, b));
            normals.push_back(point(a, b));
            create3DModel();
            return;
        }
        normals.push_back(point(bezierPoints[0].y - bezierPoints[2].y, bezierPoints[2].x - bezierPoints[0].x));
        for (int i = 0; i < bezierPoints.size() - 2; i++)
        {
            float a = bezierPoints[i].y - bezierPoints[i + 2].y;
            float b = bezierPoints[i + 2].x - bezierPoints[i].x;
            normals.push_back(point(a, b));
        }
        normals.push_back(normals[normals.size() - 1]);
        create3DModel();
        return;
    }
}

int main()
{
    // Initialize OpenGL
    if (!initOpenGL())
        return -1;

    // Initialize graphical resources.
    bool isOk = init();

    glfwSetMouseButtonCallback(g_window, mouse_button_callback);

    if (isOk)
    {
        // Main loop until window closed or escape pressed.
        while (glfwGetKey(g_window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(g_window) == 0)
        {
            for (auto& i : coordClick)
                cout << i.first << " " << i.second << endl;

            // Draw scene.
            draw();

            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();
        }
    }

    // Cleanup graphical resources.
    cleanup();

    // Tear down OpenGL.
    tearDownOpenGL();

    return isOk ? 0 : -1;
}